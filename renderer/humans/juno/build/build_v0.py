"""asset — v0 placeholder Juno from the MakeHuman/MPFB2 CC0 base (stock proportions, crude shapes, no hair).

usage: python build_v0.py <mpfb_data_dir> <out_asset_dir>
mpfb_data_dir = .../mpfb2-<ver>/src/mpfb/data
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", "..")))  # renderer/
from mh_base import MHObj, bone_matrix_asset, read_target, target_dense  # noqa: E402
from humans import asset_io  # noqa: E402

DM = 0.1  # MH decimetre -> metre


def smoothstep(a, b, x):
    t = np.clip((x - a) / (b - a), 0, 1)
    return t * t * (3 - 2 * t)


def vertex_normals(V, tris):
    fn = np.cross(V[tris[:, 1]] - V[tris[:, 0]], V[tris[:, 2]] - V[tris[:, 0]])
    n = np.zeros_like(V)
    for k in range(3):
        np.add.at(n, tris[:, k], fn)
    l = np.linalg.norm(n, axis=1, keepdims=True)
    return n / np.maximum(l, 1e-12)


def tangents(P, N, UV, tris):
    t = np.zeros_like(P)
    b = np.zeros_like(P)
    p0, p1, p2 = P[tris[:, 0]], P[tris[:, 1]], P[tris[:, 2]]
    w0, w1, w2 = UV[tris[:, 0]], UV[tris[:, 1]], UV[tris[:, 2]]
    e1, e2 = p1 - p0, p2 - p0
    d1, d2 = w1 - w0, w2 - w0
    r = d1[:, 0] * d2[:, 1] - d2[:, 0] * d1[:, 1]
    r = np.where(np.abs(r) < 1e-12, 1e-12, r)
    sd = (e1 * d2[:, 1:2] - e2 * d1[:, 1:2]) / r[:, None]
    td = (e2 * d1[:, 0:1] - e1 * d2[:, 0:1]) / r[:, None]
    for k in range(3):
        np.add.at(t, tris[:, k], sd)
        np.add.at(b, tris[:, k], td)
    t = t - N * (N * t).sum(1, keepdims=True)
    tl = np.linalg.norm(t, axis=1, keepdims=True)
    bad = tl[:, 0] < 1e-9
    alt = np.cross(N, np.array([0, 1.0, 0]))
    t[bad] = alt[bad]
    t /= np.maximum(np.linalg.norm(t, axis=1, keepdims=True), 1e-12)
    w = np.where((np.cross(N, t) * b).sum(1) < 0, -1.0, 1.0)
    return np.concatenate([t, w[:, None]], 1)


def uv_sphere(center, radius, nu=48, nv=32, fwd=np.array([0, 0, 1.0])):
    """Sphere with +z = front pole; uv = planar front projection (iris centred at 0.5,0.5); back half maps to the rim."""
    th = np.linspace(0, np.pi, nv + 1)       # 0 = front pole
    ph = np.linspace(0, 2 * np.pi, nu + 1)
    T, Ph = np.meshgrid(th, ph, indexing="ij")
    x = np.sin(T) * np.cos(Ph)
    y = np.sin(T) * np.sin(Ph)
    z = np.cos(T)
    P = np.stack([x, y, z], -1).reshape(-1, 3)
    # uv: radius in uv grows with polar angle (0..pi -> 0..0.5)
    ruv = (T / np.pi * 0.5).reshape(-1)
    U = 0.5 + ruv * np.cos(Ph.reshape(-1))
    Vv = 0.5 + ruv * np.sin(Ph.reshape(-1))
    tris = []
    W = nu + 1
    for i in range(nv):
        for j in range(nu):
            a, b, c, d = i * W + j, i * W + j + 1, (i + 1) * W + j, (i + 1) * W + j + 1
            if i > 0:
                tris.append([a, c, b])
            if i < nv - 1:
                tris.append([b, c, d])
    tris = np.array(tris, np.int64)
    # orientation: front pole +z already; ensure CCW outward
    return center + radius * P, P.copy(), np.stack([U, Vv], 1), tris


def main(data, out):
    obj = MHObj(os.path.join(data, "3dobjs", "base.obj"))
    nV = len(obj.v)
    V = obj.v.copy()
    T = os.path.join(data, "targets")
    # stock young woman (MH default ethnic mix), ideal proportions
    for eth in ("african", "asian", "caucasian"):
        V += target_dense(os.path.join(T, "macrodetails", f"{eth}-female-young.target.gz"), nV) / 3.0
    p = os.path.join(T, "macrodetails", "proportions", "female-young-averagemuscle-averageweight-idealproportions.target.gz")
    if os.path.exists(p):
        V += target_dense(p, nV)
    # a little animated-film push for v0: bigger eyes
    for side in ("l", "r"):
        V += 0.6 * target_dense(os.path.join(T, "eyes", f"{side}-eye-scale-incr.target.gz"), nV)
    body_v = obj.group_verts("body")
    floor = V[body_v, 1].min()
    V[:, 1] -= floor
    VA = V * DM  # asset metres

    def cube(name):
        vs = obj.group_verts(name)
        return VA[vs].mean(0)

    # ---------------- skeleton ----------------
    rig = json.load(open(os.path.join(data, "rigs", "standard", "rig.game_engine.json")))

    def endpoint(spec):
        if spec["strategy"] == "CUBE":
            return cube(spec["cube_name"])
        return VA[np.array(spec["vertex_indices"])].mean(0)

    bones = {}
    for name, b in rig.items():
        nm = "root" if name == "Root" else name
        par = b["parent"]
        par = "root" if par == "Root" else (par or None)
        bones[nm] = dict(parent=par, head=endpoint(b["head"]), tail=endpoint(b["tail"]), roll=b.get("roll", 0.0))
    bones["root"]["head"] = np.array([0.0, 0.0, 0.0])
    bones["root"]["tail"] = np.array([0.0, 0.0, -0.1])  # Blender: +y_b points forward(-y_b)? keep a fixed axis
    hj = cube("joint-mouth")      # MH: joint-mouth = jaw hinge, joint-jaw = chin
    chin = cube("joint-jaw")
    chin_dir = chin - hj
    bones["jaw"] = dict(parent="head", head=hj, tail=chin, roll=0.0)
    eye_c = {}
    for side in ("l", "r"):
        ev = obj.group_verts(f"helper-{side}-eye")
        c = VA[ev].mean(0)
        r = np.linalg.norm(VA[ev] - c, axis=1).mean()
        eye_c[side] = (c, r)
        bones[f"eye_{side}"] = dict(parent="head", head=c, tail=c + np.array([0, 0, 0.02]), roll=0.0)
    # topological order
    order = []
    def visit(n):
        if n in order:
            return
        p = bones[n]["parent"]
        if p:
            visit(p)
        order.append(n)
    for n in ["root"] + sorted(k for k in bones if k != "root"):
        visit(n)
    bidx = {n: i for i, n in enumerate(order)}
    rest = np.stack([bone_matrix_asset(bones[n]["head"], bones[n]["tail"], bones[n]["roll"]) for n in order])
    skel = {"bones": [{"name": n, "parent": bidx[bones[n]["parent"]] if bones[n]["parent"] else -1,
                        "rest": rest[i].reshape(-1).tolist(), "inverse_bind": np.linalg.inv(rest[i]).reshape(-1).tolist(),
                        "helper": False, "length": float(np.linalg.norm(bones[n]["tail"] - bones[n]["head"]))}
                       for i, n in enumerate(order)]}

    # ---------------- weights (MH game-engine, CC0) ----------------
    wj = json.load(open(os.path.join(data, "rigs", "standard", "weights.game_engine.json")))["weights"]
    per = [[] for _ in range(nV)]
    for bn, lst in wj.items():
        b = bidx["root" if bn == "Root" else bn]
        for vi, w in lst:
            per[vi].append((w, b))
    for vi in obj.group_verts("helper-lower-teeth").tolist() + obj.group_verts("helper-tongue").tolist():
        per[vi] = [(1.0, bidx["jaw"])]
    for vi in obj.group_verts("helper-upper-teeth").tolist():
        per[vi] = [(1.0, bidx["head"])]
    J = np.zeros((nV, 4), np.uint16)
    W = np.zeros((nV, 4), np.float32)
    for vi in range(nV):
        l = sorted(per[vi], reverse=True)[:4] or [(1.0, bidx["root"])]
        s = sum(w for w, _ in l)
        for k, (w, b) in enumerate(l):
            J[vi, k] = b
            W[vi, k] = w / s

    # ---------------- face shapes (MH expression units, CC0) ----------------
    U = os.path.join(T, "expression", "units", "caucasian")
    skin = np.zeros(nV, bool)
    skin[body_v] = True

    def ut(name):
        d = target_dense(os.path.join(U, name + ".target.gz"), nV) * DM
        d[~skin] = 0
        return d

    x = VA[:, 0]
    sideL = smoothstep(-0.006, 0.006, x)   # anatomical left = +x
    sideR = 1 - sideL
    cmag = np.linalg.norm(ut("mouth-compression"), axis=1) ** 2
    mo = (VA * cmag[:, None]).sum(0) / cmag.sum()   # lip centre
    mo[0] = 0.0
    mouth_mask = np.zeros(nV)
    for n in ("mouth-open", "mouth-compression", "mouth-pursing", "mouth-corner-puller", "mouth-depression"):
        mouth_mask = np.maximum(mouth_mask, np.linalg.norm(ut(n), axis=1))
    mouth_mask = np.clip(mouth_mask / max(mouth_mask.max(), 1e-9) * 3, 0, 1)
    upper = smoothstep(-0.003, 0.003, VA[:, 1] - mo[1])
    lower = 1 - upper
    near_mouth = np.exp(-((VA - mo) ** 2).sum(1) / (2 * 0.012 ** 2)) * skin

    S = {}
    S["eyeBlinkLeft"] = ut("eye-left-closure"); S["eyeBlinkRight"] = ut("eye-right-closure")
    S["eyeWideLeft"] = ut("eye-left-opened-up"); S["eyeWideRight"] = ut("eye-right-opened-up")
    S["eyeSquintLeft"] = ut("eye-left-slit"); S["eyeSquintRight"] = ut("eye-right-slit")
    for side, eye, sgn in (("Left", "l", 1), ("Right", "r", -1)):
        lid = np.linalg.norm(ut(f"eye-{'left' if eye == 'l' else 'right'}-closure"), axis=1)
        lid = np.clip(lid / max(lid.max(), 1e-9) * 2, 0, 1)[:, None]
        S[f"eyeLookUp{side}"] = lid * np.array([0, 0.0015, 0])
        S[f"eyeLookDown{side}"] = lid * np.array([0, -0.0020, 0])
        S[f"eyeLookIn{side}"] = lid * np.array([-sgn * 0.0008, 0, 0])
        S[f"eyeLookOut{side}"] = lid * np.array([sgn * 0.0008, 0, 0])
        below = (VA[:, 1] < eye_c[eye][0][1]).astype(float)[:, None]
        sl = ut(f"eye-{'left' if eye == 'l' else 'right'}-slit")
        S[f"cheekSquint{side}"] = sl * below * 0.6
        S[f"eyeHappy{side}"] = lid * below * np.array([0, 0.0030, 0.0005])
    S["browDownLeft"] = ut("eyebrows-left-down"); S["browDownRight"] = ut("eyebrows-right-down")
    S["browOuterUpLeft"] = ut("eyebrows-left-extern-up"); S["browOuterUpRight"] = ut("eyebrows-right-extern-up")
    S["browInnerUp"] = 0.5 * (ut("eyebrows-left-inner-up") + ut("eyebrows-right-inner-up"))
    S["browTiltLeft"] = ut("eyebrows-left-inner-up") - 0.5 * ut("eyebrows-left-extern-up")
    S["browTiltRight"] = ut("eyebrows-right-inner-up") - 0.5 * ut("eyebrows-right-extern-up")
    S["jawOpen"] = ut("mouth-open")
    jm = np.linalg.norm(S["jawOpen"], axis=1)
    jaw_region = np.clip(jm / max(jm.max(), 1e-9) * 2, 0, 1)[:, None] * (VA[:, 1] < mo[1]).astype(float)[:, None]
    S["jawForward"] = jaw_region * np.array([0, 0, 0.004])
    S["jawLeft"] = jaw_region * np.array([0.004, 0, 0])
    S["jawRight"] = jaw_region * np.array([-0.004, 0, 0])
    cp = ut("mouth-corner-puller")
    S["mouthSmileLeft"] = cp * sideL[:, None]; S["mouthSmileRight"] = cp * sideR[:, None]
    dp = ut("mouth-depression")
    S["mouthFrownLeft"] = dp * sideL[:, None]; S["mouthFrownRight"] = dp * sideR[:, None]
    rt = ut("mouth-retraction")
    S["mouthStretchLeft"] = rt * sideL[:, None]; S["mouthStretchRight"] = rt * sideR[:, None]
    S["mouthDimpleLeft"] = 0.5 * rt * sideL[:, None]; S["mouthDimpleRight"] = 0.5 * rt * sideR[:, None]
    S["mouthPucker"] = ut("mouth-pursing")
    S["mouthFunnel"] = ut("mouth-protusion")
    cm = ut("mouth-compression")
    S["mouthPressLeft"] = cm * sideL[:, None]; S["mouthPressRight"] = cm * sideR[:, None]
    S["mouthClose"] = cm.copy()
    S["lipsTogether"] = 0.5 * cm
    S["mouthRollLower"] = cm * lower[:, None]; S["mouthRollUpper"] = cm * upper[:, None]
    el = ut("mouth-elevation")
    S["mouthShrugUpper"] = el * upper[:, None]; S["mouthShrugLower"] = el * lower[:, None]
    ur = ut("mouth-upward-retraction")
    S["mouthUpperUpLeft"] = ur * sideL[:, None]; S["mouthUpperUpRight"] = ur * sideR[:, None]
    dr = ut("mouth-depression-retraction")
    S["mouthLowerDownLeft"] = dr * sideL[:, None]; S["mouthLowerDownRight"] = dr * sideR[:, None]
    S["mouthLeft"] = mouth_mask[:, None] * np.array([0.004, 0, 0])
    S["mouthRight"] = mouth_mask[:, None] * np.array([-0.004, 0, 0])
    S["noseSneerLeft"] = ut("nose-left-elevation"); S["noseSneerRight"] = ut("nose-right-elevation")
    Nb = vertex_normals(VA, np.array([t for f in obj.faces for t in ([f[0][0], f[1][0], f[2][0]],) + (([f[0][0], f[2][0], f[3][0]],) if len(f) == 4 else ())]))
    cheek = np.zeros(nV)
    for sx in (1, -1):
        cc = mo + np.array([sx * 0.028, 0.006, -0.012])
        cheek += np.exp(-((VA - cc) ** 2).sum(1) / (2 * 0.013 ** 2))
    S["cheekPuff"] = (cheek * skin)[:, None] * Nb * 0.004
    S["labioDental"] = (mouth_mask * lower * near_mouth)[:, None] * np.array([0, 0.002, -0.003])
    S["lipTuckLower"] = (mouth_mask * lower * near_mouth)[:, None] * np.array([0, 0.003, -0.004])
    S["lipTuckUpper"] = (mouth_mask * upper * near_mouth)[:, None] * np.array([0, -0.002, -0.004])
    tv = obj.group_verts("helper-tongue")
    tz = VA[tv, 2]
    tip = np.zeros(nV)
    tip[tv] = smoothstep(tz.min(), tz.max(), tz)
    S["tongueOut"] = (tip ** 1.5)[:, None] * np.array([0, -0.006, 0.030])
    S["tongueTipUp"] = (tip ** 2)[:, None] * np.array([0, 0.008, 0.002])
    S["tongueDental"] = (tip ** 2)[:, None] * np.array([0, 0.002, 0.010])
    back = np.zeros(nV)
    back[tv] = 1 - tip[tv]
    S["tongueBack"] = (back ** 2)[:, None] * np.array([0, 0.006, 0])
    missing = [s for s in asset_io.REQUIRED_SHAPES if s not in S]
    assert not missing, missing

    # jaw open angle that matches the mouth-open target at the chin
    chin_vs = np.where((jm > 0.6 * jm.max()) & skin)[0]
    a0 = VA[chin_vs] - hj
    a1 = a0 + S["jawOpen"][chin_vs]
    ang = np.degrees(np.angle(np.exp(1j * (np.arctan2(a1[:, 1], a1[:, 2]) - np.arctan2(a0[:, 1], a0[:, 2])))))
    jaw_deg = float(-np.median(ang))  # positive = opening (chin down)

    # ---------------- submeshes ----------------
    tri_of = lambda f: [[f[0], f[1], f[2]]] + ([[f[0], f[2], f[3]]] if len(f) == 4 else [])
    head_w = np.zeros(nV)
    for vi in range(nV):
        for k in range(4):
            if J[vi, k] in (bidx["head"], bidx["neck_01"], bidx["jaw"]):
                head_w[vi] += W[vi, k]
    sub_faces = {"head": [], "body": [], "teeth_upper": [], "teeth_lower": [], "tongue": [], "lashes_l": [], "lashes_r": [],
                 "cloth_suit": []}
    for fi, f in enumerate(obj.faces):
        g = obj.fgroup[fi]
        if g == "body":
            hw = np.mean([head_w[vi] for vi, _ in f])
            sub_faces["head" if hw > 0.5 else "body"].append(f)
        elif g == "helper-upper-teeth":
            sub_faces["teeth_upper"].append(f)
        elif g == "helper-lower-teeth":
            sub_faces["teeth_lower"].append(f)
        elif g == "helper-tongue":
            sub_faces["tongue"].append(f)
        elif g and g.startswith("helper-l-eyelashes"):
            sub_faces["lashes_l"].append(f)
        elif g and g.startswith("helper-r-eyelashes"):
            sub_faces["lashes_r"].append(f)
        elif g == "helper-tights":
            sub_faces["cloth_suit"].append(f)
    mats = {"head": "skin_head", "body": "skin_body", "teeth_upper": "teeth", "teeth_lower": "teeth", "tongue": "tongue",
            "lashes_l": "lashes", "lashes_r": "lashes", "cloth_suit": "cloth_knit"}

    # smooth normals on the welded MH mesh (per original vertex)
    all_tris = np.array([t for f in obj.faces for t in tri_of([vi for vi, _ in f])], np.int64)
    Nw = vertex_normals(VA, all_tris)
    cloth_v = obj.group_verts("helper-tights")
    VAc = VA.copy()
    VAc[cloth_v] += Nw[cloth_v] * 0.004  # v0: suit floats 4 mm off the skin

    P_l, N_l, UV_l, J_l, W_l, I_l, orig_l, subs = [], [], [], [], [], [], [], []
    base = 0
    idx_base = 0
    for name in ["body", "head", "teeth_upper", "teeth_lower", "tongue", "cloth_suit", "lashes_l", "lashes_r"]:
        fl = sub_faces[name]
        keymap, orig, uvs, tris = {}, [], [], []
        for f in fl:
            loc = []
            for vi, ti in f:
                k = (vi, ti)
                if k not in keymap:
                    keymap[k] = len(orig)
                    orig.append(vi)
                    uvs.append(obj.vt[ti] if ti >= 0 else (0.0, 0.0))
                loc.append(keymap[k])
            tris += tri_of(loc)
        orig = np.array(orig, np.int64)
        tris = np.array(tris, np.int64)
        P = VAc[orig]
        Nn = Nw[orig]
        UVa = np.array(uvs, np.float64)
        P_l.append(P); N_l.append(Nn); UV_l.append(UVa); J_l.append(J[orig]); W_l.append(W[orig]); orig_l.append(orig)
        I_l.append(tris + base)
        blend = "alpha" if name.startswith("lashes") else "opaque"
        subs.append(dict(name=name, material=mats[name], first_index=idx_base, index_count=int(tris.size),
                         first_vertex=base, vertex_count=int(len(orig)), blend=blend,
                         double_sided=name.startswith("lashes"), shapes=name in ("head", "tongue", "body")))
        base += len(orig)
        idx_base += tris.size
    # procedural eyes + corneas
    for side in ("l", "r"):
        c, r = eye_c[side]
        for nm, rad, mat, blend in ((f"eye_{side}", r, f"eye_{side}", "opaque"), (f"cornea_{side}", r * 1.03, "cornea", "alpha")):
            P, Nn, UVa, tris = uv_sphere(c, rad)
            if nm.startswith("cornea"):
                # cornea bulge over the iris (front cap)
                fz = Nn[:, 2]
                P = P + Nn * (np.clip((fz - 0.80) / 0.20, 0, 1) ** 1.5 * 0.12 * rad)[:, None]
            n = len(P)
            P_l.append(P); N_l.append(Nn); UV_l.append(UVa)
            J_l.append(np.tile(np.array([bidx[f"eye_{side}"], 0, 0, 0], np.uint16), (n, 1)))
            W_l.append(np.tile(np.array([1, 0, 0, 0], np.float32), (n, 1)))
            orig_l.append(np.full(n, -1, np.int64))
            I_l.append(tris + base)
            subs.append(dict(name=nm, material=mat, first_index=idx_base, index_count=int(tris.size), first_vertex=base,
                             vertex_count=n, blend=blend, double_sided=False, shapes=False))
            base += n
            idx_base += tris.size
    P = np.concatenate(P_l); Nn = np.concatenate(N_l); UVa = np.concatenate(UV_l)
    I = np.concatenate([i.reshape(-1) for i in I_l]).astype(np.uint32)
    Jg = np.concatenate(J_l); Wg = np.concatenate(W_l); orig = np.concatenate(orig_l)
    Tg = tangents(P, Nn, UVa, I.reshape(-1, 3).astype(np.int64))

    # shapes -> split vertices; normal deltas by recompute on the welded mesh
    names = asset_io.REQUIRED_SHAPES
    shapes = []
    valid = orig >= 0
    for nm in names:
        d = S[nm]
        Nm = vertex_normals(VA + d, all_tris)
        dn = Nm - Nw
        moved = np.where(np.linalg.norm(d, axis=1) > 1e-7)[0]
        mv = np.zeros(nV, bool)
        mv[moved] = True
        sel = np.where(valid & mv[np.where(valid, orig, 0)])[0]
        shapes.append((nm, sel, d[orig[sel]], dn[orig[sel]]))

    os.makedirs(out, exist_ok=True)
    asset_io.write_mesh(os.path.join(out, "lod0"), dict(position=P, normal=Nn, tangent=Tg, uv0=UVa, joints=Jg, weights=Wg,
                                                         index=I), subs, shapes)
    # lod1 = same mesh in v0
    import shutil
    shutil.copytree(os.path.join(out, "lod0"), os.path.join(out, "lod1"), dirs_exist_ok=True)
    skel["skeleton_hints"] = {"jaw_open_deg": round(jaw_deg, 2), "jaw_open_axis": [1, 0, 0],
                              "jaw_side_deg": 4.0, "jaw_forward_m": 0.003,
                              "eye_center_l": eye_c["l"][0].round(5).tolist(), "eye_center_r": eye_c["r"][0].round(5).tolist(),
                              "eye_radius_m": round(float(eye_c["l"][1]), 5)}
    with open(os.path.join(out, "skeleton.json"), "w") as f:
        json.dump(skel, f, indent=0)

    # ---------------- textures (v0: flat-ish) ----------------
    td = os.path.join(out, "tex")
    os.makedirs(td, exist_ok=True)
    rng = np.random.default_rng(7)

    def save(name, arr):
        im = Image.fromarray(arr.astype(np.uint8))
        im.save(os.path.join(td, name + ".png"))
        im.resize((max(1, arr.shape[1] // 2), max(1, arr.shape[0] // 2)), Image.LANCZOS).save(os.path.join(td, name + "_1k.png"))

    def flat(rgb, n=1024, noise=6):
        a = np.ones((n, n, 4), np.float32) * 255
        a[..., :3] = np.array(rgb, np.float32) + rng.normal(0, noise, (n, n, 1))
        return np.clip(a, 0, 255)

    def flat_n(n=512):
        a = np.zeros((n, n, 3), np.uint8); a[...] = (128, 128, 255); return a

    def packed(ao=255, rough=0.5, spec=0.5, thick=0.0, n=512):
        a = np.zeros((n, n, 4), np.uint8); a[...] = (ao, int(rough * 255), int(spec * 255), int(thick * 255)); return a

    mats_json = {}
    for m, rgb, rough, thick, typ in (("skin_head", (214, 160, 125), 0.45, 0.3, "skin"), ("skin_body", (214, 160, 125), 0.5, 0.2, "skin"),
                                      ("teeth", (238, 232, 218), 0.25, 0.3, "teeth"), ("tongue", (200, 95, 100), 0.35, 0.5, "tongue"),
                                      ("lashes", (30, 20, 16), 0.6, 0.0, "lashes"), ("cloth_knit", (196, 142, 52), 0.85, 0.0, "cloth")):
        save(f"{m}_albedo", flat(rgb, 1024 if m.startswith("skin") else 256))
        save(f"{m}_normal", flat_n())
        save(f"{m}_packed", packed(rough=rough, thick=thick))
        e = {"type": typ, "albedo": f"tex/{m}_albedo.png", "normal": f"tex/{m}_normal.png", "packed": f"tex/{m}_packed.png",
             "params": {}}
        if typ == "skin":
            e["params"] = {"sss_color": [0.9, 0.35, 0.25], "sss_radius_mm": [3.6, 1.4, 0.8], "spec_f0": 0.028}
        if typ == "cloth":
            e["params"] = {"sheen_color": [0.9, 0.75, 0.5], "sheen_roughness": 0.6}
        mats_json[m] = e
    # eye texture: sclera + amber iris + pupil, planar uv (iris radius 0.5*theta/pi)
    n = 1024
    yy, xx = np.mgrid[0:n, 0:n]
    u = (xx + 0.5) / n - 0.5
    v = (yy + 0.5) / n - 0.5
    rr = np.sqrt(u * u + v * v)
    ang = np.arctan2(v, u)
    iris_r = 0.5 * np.radians(36) / np.pi   # iris ~ 36 deg half-angle on the ball
    pupil_r = iris_r * 0.38
    col = np.zeros((n, n, 3), np.float32)
    col[...] = (236, 230, 226)
    t = rr / iris_r
    fib = 0.85 + 0.15 * np.sin(ang * 90 + rng.normal(0, 1, (n, n)) * 0.3)
    iris = np.stack([150 - 60 * t, 92 - 45 * t, 40 - 20 * t], -1) * fib[..., None]
    m_iris = rr < iris_r
    col[m_iris] = iris[m_iris]
    limb = smoothstep(iris_r * 0.85, iris_r * 1.02, rr) * (rr < iris_r * 1.08)
    col = col * (1 - 0.6 * limb[..., None])
    col[rr < pupil_r] = (8, 6, 6)
    a = np.concatenate([col, np.full((n, n, 1), 255.0)], -1)
    for side in ("l", "r"):
        save(f"eye_{side}_albedo", np.clip(a, 0, 255))
        save(f"eye_{side}_normal", flat_n())
        save(f"eye_{side}_packed", packed(rough=0.15))
        mats_json[f"eye_{side}"] = {"type": "eye", "albedo": f"tex/eye_{side}_albedo.png", "normal": f"tex/eye_{side}_normal.png",
                                    "packed": f"tex/eye_{side}_packed.png",
                                    "params": {"iris_uv_radius": float(iris_r), "pupil_uv_radius": float(pupil_r), "limbus_width": 0.08}}
    mats_json["cornea"] = {"type": "cornea", "params": {"ior": 1.376, "roughness": 0.02}}

    allP = P
    height = float(allP[np.concatenate([np.arange(s["first_vertex"], s["first_vertex"] + s["vertex_count"])
                                        for s in subs if s["name"] in ("head", "body")]), 1].max())

    def grasp(side):
        tipT = cube(f"joint-{side}-finger-1-4")
        midP = cube(f"joint-{side}-finger-3-2")
        o = 0.5 * (tipT + midP)
        Hm = rest[bidx[f"hand_{side}"]]
        G = Hm.copy()
        G[:3, 3] = o
        return {"bone": f"hand_{side}", "offset": (np.linalg.inv(Hm) @ G).reshape(-1).round(6).tolist()}

    meta = {"format": asset_io.FORMAT, "name": "juno", "version": "v0-placeholder", "units": "m", "up": "+y", "forward": "+z",
            "height_m": round(height, 4), "scale_to_H": round(1 / height, 5),
            "bounds": {"min": allP.min(0).round(4).tolist(), "max": allP.max(0).round(4).tolist()},
            "rest_pose": "A", "skeleton": "skeleton.json",
            "lods": [{"name": "lod0", "mesh": "lod0/mesh.json", "tex_suffix": "", "hair": {}},
                     {"name": "lod1", "mesh": "lod1/mesh.json", "tex_suffix": "_1k", "hair": {}}],
            "shapes": list(names), "materials": mats_json, "hair_groups": {},
            "landmarks": {"eye_l": eye_c["l"][0].round(5).tolist(), "eye_r": eye_c["r"][0].round(5).tolist(),
                          "mouth": mo.round(5).tolist(), "head_top": [0.0, round(height, 4), 0.0],
                          "chin": (hj + chin_dir).round(5).tolist(), "grasp_l": grasp("l"), "grasp_r": grasp("r")},
            "notes": "v0 placeholder: MakeHuman/MPFB2 CC0 base, stock young-woman proportions, crude face shapes from MH "
                     "expression units + procedural extras, a body suit instead of the wardrobe, no hair. Grasp frame "
                     "orientation = hand bone axes in v0."}
    with open(os.path.join(out, "juno.asset.json"), "w") as f:
        json.dump(meta, f, indent=1)
    with open(os.path.join(out, "LICENSES.md"), "w") as f:
        f.write("# juno.asset v0 — sources\n\n| what | source | licence | changes |\n|---|---|---|---|\n"
                "| base mesh hm08 (body, teeth, tongue, lash cards, tights helper), macro/eye/expression-unit targets, "
                "game_engine rig + weights | MPFB2 v2.0.18 data, https://github.com/makehumancommunity/mpfb2 "
                "(src/mpfb/data) | CC0 1.0 (LICENSE.ASSETS.md) | targets applied, re-rigged to the bone set "
                "(+jaw, eye_l/r), re-weighted teeth/tongue, triangulated, UV-split, exported to juno.asset/1 |\n"
                "| eyes, corneas, all textures | generated by renderer/humans/juno/build/build_v0.py | our own work | — |\n")
    print(json.dumps({"height_m": height, "jaw_open_deg": jaw_deg, "verts": len(P), "tris": len(I) // 3}))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
