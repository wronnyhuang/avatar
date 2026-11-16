"""asset — Part: one exported submesh, plus helpers to cut Parts out of the MH base and to export the asset.

A Part holds LOCAL arrays (its own vertex list). Shapes on a Part: dict name -> (local idx (K,), dpos (K,3)); normals for
shapes are recomputed by the exporter from the Part's own triangles (plus `weld` ids so UV seams don't crack).

Module contract (every build module returns these; see build_juno.py):
  face_shapes.build(base)                 -> {shape_name: (nV,3) dpos on MH vertices}  (all 63 names, skin + teeth/tongue)
  eyes_mouth.build(base, texdir)          -> (parts: [Part], materials: {id: material json}, hide_faces: set(MH face ids))
  skin_tex.build(base, texdir)            -> materials {skin_head, skin_body, ...}  (+ writes PNGs into texdir)
  hair.build(base, outdir)                -> {group: hair_group json incl. "lod": {lod0: file, lod1: file}} (+ writes .bin)
  wardrobe.build(base, texdir)            -> (parts: [Part], materials: {...}, hide_faces: set(MH body face ids))
"""
from __future__ import annotations

import json
import os
import shutil
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", "..")))
from juno_base import tri_of, vertex_normals  # noqa: E402
from humans import asset_io  # noqa: E402


class Part:
    def __init__(self, name, material, P, UV, tris, J, W, N=None, shapes=None, weld=None, blend="opaque",
                 double_sided=False, orig=None):
        self.name, self.material = name, material
        self.P = np.asarray(P, np.float64)
        self.UV = np.asarray(UV, np.float64)
        self.tris = np.asarray(tris, np.int64).reshape(-1, 3)
        self.J = np.asarray(J, np.uint16)
        self.W = np.asarray(W, np.float32)
        self.weld = np.arange(len(self.P)) if weld is None else np.asarray(weld, np.int64)  # same id = same point
        self.N = N
        self.shapes = shapes or {}
        self.blend, self.double_sided = blend, double_sided
        self.orig = orig  # MH vertex id per local vertex (or None)

    def welded_normals(self, P=None):
        P = self.P if P is None else P
        u, inv = np.unique(self.weld, return_inverse=True)
        Pw = np.zeros((len(u), 3))
        Pw[inv] = P
        n = vertex_normals(Pw, inv[self.tris])
        return n[inv]


def mh_part(base, name, material, faces, VA=None, shapes_mh=None, J=None, W=None, **kw):
    """Cut a Part from MH faces (list of MH face ids), splitting vertices at UV seams. shapes_mh: {name: (nV,3)}."""
    VA = base.VA if VA is None else VA
    J = base.J if J is None else J
    W = base.W if W is None else W
    keymap, orig, uvs, tris = {}, [], [], []
    for fi in faces:
        f = base.obj.faces[fi]
        loc = []
        for vi, ti in f:
            k = (vi, ti)
            if k not in keymap:
                keymap[k] = len(orig)
                orig.append(vi)
                uvs.append(base.obj.vt[ti] if ti >= 0 else (0.0, 0.0))
            loc.append(keymap[k])
        tris += tri_of(loc)
    orig = np.array(orig, np.int64)
    shapes = {}
    for sn, d in (shapes_mh or {}).items():
        dl = d[orig]
        sel = np.where(np.linalg.norm(dl, axis=1) > 1e-8)[0]
        if len(sel):
            shapes[sn] = (sel, dl[sel])
    return Part(name, material, VA[orig], np.array(uvs), np.array(tris), J[orig], W[orig], shapes=shapes, weld=orig,
                orig=orig, **kw)


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
    bad = np.linalg.norm(t, axis=1) < 1e-9
    t[bad] = np.cross(N, np.array([0, 1.0, 0]))[bad]
    t /= np.maximum(np.linalg.norm(t, axis=1, keepdims=True), 1e-12)
    w = np.where((np.cross(N, t) * b).sum(1) < 0, -1.0, 1.0)
    return np.concatenate([t, w[:, None]], 1)


def export(out_dir, lod, parts, shape_names, shape_meta=None):
    """Write <out_dir>/<lod>/mesh.json|bin|shapes.bin. Returns submesh list."""
    P_l, N_l, T_l, UV_l, J_l, W_l, I_l, subs = [], [], [], [], [], [], [], []
    acc = {s: ([], [], []) for s in shape_names}
    base = idx = 0
    for p in parts:
        N = p.welded_normals() if p.N is None else np.asarray(p.N, np.float64)
        T = tangents(p.P, N, p.UV, p.tris)
        P_l.append(p.P); N_l.append(N); T_l.append(T); UV_l.append(p.UV); J_l.append(p.J); W_l.append(p.W)
        I_l.append((p.tris + base).reshape(-1))
        for sn, (li, dp) in p.shapes.items():
            if sn not in acc:
                raise KeyError(f"part {p.name}: shape {sn} not in shape list")
            Pm = p.P.copy()
            Pm[li] += dp
            Nm = p.welded_normals(Pm)
            dn = (Nm - N)
            moved = np.zeros(len(p.P), bool)
            moved[li] = True
            # asset-smile2: a product corrective's normal delta is taken IN CONTEXT (its drivers at 1), so
            # N + dn_a + dn_b + dn_corr is the exact normal at a = b = 1. Taken alone (the corrective added to the rest
            # mesh) it was garbage where the corrective is large: the sticky-lip corr_jawOpen_jawOpen lit the lower lip
            # lining as a bright jagged crescent in real-time at jawOpen 1 (positions fine, normals wrong).
            of = ((shape_meta or {}).get(sn, {}).get("corrective") or {}).get("of")
            if of:
                drv = []
                for a in dict.fromkeys(of):
                    if a in p.shapes:
                        drv.append(p.shapes[a])
                Pa = p.P.copy()
                for la, da in drv:
                    Pa[la] += da
                    moved[la] = True
                Nctx = p.welded_normals(Pa + (Pm - p.P))
                dn = Nctx - p.welded_normals(Pa)
                if len(drv) == 2:
                    # both drivers at 1: N(a+b+c) - N(a+b) + [N(a+b) - N(a) - N(b) + N] (the drivers' cross term)
                    Pa1, Pb1 = p.P.copy(), p.P.copy()
                    Pa1[drv[0][0]] += drv[0][1]
                    Pb1[drv[1][0]] += drv[1][1]
                    dn = dn + p.welded_normals(Pa) - p.welded_normals(Pa1) - p.welded_normals(Pb1) + N
            # normals change around moved vertices too: include 1-ring
            ring = np.zeros(len(p.P), bool)
            tm = moved[p.tris].any(1)
            ring[p.tris[tm].reshape(-1)] = True
            sel = np.where(ring)[0]
            full = np.zeros_like(p.P)
            full[li] = dp
            acc[sn][0].append(sel + base); acc[sn][1].append(full[sel]); acc[sn][2].append(dn[sel])
        subs.append(dict(name=p.name, material=p.material, first_index=idx, index_count=int(p.tris.size),
                         first_vertex=base, vertex_count=int(len(p.P)), blend=p.blend, double_sided=p.double_sided,
                         shapes=bool(p.shapes)))
        base += len(p.P)
        idx += p.tris.size
    shapes = []
    for sn in shape_names:
        ii, dp, dn = acc[sn]
        extra = (shape_meta or {}).get(sn, {})
        if ii:
            shapes.append((sn, np.concatenate(ii), np.concatenate(dp), np.concatenate(dn), extra))
        else:
            shapes.append((sn, np.zeros(0, np.uint32), np.zeros((0, 3)), np.zeros((0, 3)), extra))
    W = np.concatenate(W_l)
    W = W / np.maximum(W.sum(1, keepdims=True), 1e-9)
    asset_io.write_mesh(os.path.join(out_dir, lod), dict(
        position=np.concatenate(P_l), normal=np.concatenate(N_l), tangent=np.concatenate(T_l), uv0=np.concatenate(UV_l),
        joints=np.concatenate(J_l), weights=W, index=np.concatenate(I_l).astype(np.uint32)), subs, shapes)
    return subs


def catmull_clark(faces, nV):
    """faces: list of lists of vertex ids (quads/tris, MH ids). Returns (S sparse (n_new, nV), new_faces list of
    4-tuples of NEW ids, corner_src: per new face 4 tuples describing each corner as ('v', vi) | ('e', a, b) | ('f', fi)
    so face-varying data (UVs) can be interpolated. New ids: [vertex points for used verts][edge points][face points]."""
    import scipy.sparse as sp
    used = np.unique(np.concatenate([np.asarray(f) for f in faces]))
    vloc = -np.ones(nV, np.int64)
    vloc[used] = np.arange(len(used))
    nv = len(used)
    edge_id, edge_faces = {}, []
    face_edges = []
    for fi, f in enumerate(faces):
        es = []
        for i in range(len(f)):
            a, b = f[i], f[(i + 1) % len(f)]
            k = (a, b) if a < b else (b, a)
            if k not in edge_id:
                edge_id[k] = len(edge_faces)
                edge_faces.append([])
            edge_faces[edge_id[k]].append(fi)
            es.append(edge_id[k])
        face_edges.append(es)
    ne, nf = len(edge_faces), len(faces)
    rows, cols, vals = [], [], []
    # face points
    fp_rows = []
    for fi, f in enumerate(faces):
        r = nv + ne + fi
        for v in f:
            rows.append(r); cols.append(v); vals.append(1.0 / len(f))
    edges = [None] * ne
    for k, e in edge_id.items():
        edges[e] = k
    boundary = np.array([len(fl) == 1 for fl in edge_faces])
    for e, (a, b) in enumerate(edges):
        r = nv + e
        if boundary[e] or len(edge_faces[e]) != 2:
            rows += [r, r]; cols += [a, b]; vals += [0.5, 0.5]
        else:
            rows += [r, r]; cols += [a, b]; vals += [0.25, 0.25]
            for fi in edge_faces[e]:
                for v in faces[fi]:
                    rows.append(r); cols.append(v); vals.append(0.25 / len(faces[fi]))
    # vertex points
    v_edges = [[] for _ in range(nv)]
    v_faces = [[] for _ in range(nv)]
    for e, (a, b) in enumerate(edges):
        v_edges[vloc[a]].append(e); v_edges[vloc[b]].append(e)
    for fi, f in enumerate(faces):
        for v in f:
            v_faces[vloc[v]].append(fi)
    for li, v in enumerate(used):
        r = li
        be = [e for e in v_edges[li] if boundary[e]]
        if be:
            if len(be) == 2:
                rows.append(r); cols.append(v); vals.append(0.75)
                for e in be:
                    a, b = edges[e]
                    rows.append(r); cols.append(b if a == v else a); vals.append(0.125)
            else:
                rows.append(r); cols.append(v); vals.append(1.0)
            continue
        n = len(v_edges[li])
        fl = v_faces[li]
        # Q/n : average of face points
        for fi in fl:
            for u in faces[fi]:
                rows.append(r); cols.append(u); vals.append(1.0 / len(fl) / len(faces[fi]) / n)
        # 2R/n : average of edge midpoints
        for e in v_edges[li]:
            a, b = edges[e]
            rows += [r, r]; cols += [a, b]; vals += [2.0 / n / len(v_edges[li]) * 0.5] * 2
        rows.append(r); cols.append(v); vals.append((n - 3.0) / n)
    S = sp.coo_matrix((vals, (rows, cols)), shape=(nv + ne + nf, nV)).tocsr()
    new_faces, corner_src = [], []
    for fi, f in enumerate(faces):
        k = len(f)
        es = face_edges[fi]
        for i in range(k):
            ids = (vloc[f[i]], nv + es[i], nv + ne + fi, nv + es[i - 1])
            src = (("v", i), ("e", i, (i + 1) % k), ("f",), ("e", (i - 1) % k, i))
            new_faces.append(ids)
            corner_src.append((fi, src))
    return S, new_faces, corner_src


def mh_subdiv_parts(base, groups, shapes_mh=None, levels=1, VA=None, J=None, W=None):
    """Catmull-Clark the union of several MH face sets (so shared borders match), then split back into Parts.
    groups: list of (name, material, faces). Face-varying UVs are interpolated linearly. Weights -> dense -> top 4."""
    VA = base.VA if VA is None else VA
    J = base.J if J is None else J
    W = base.W if W is None else W
    nb = len(base.bone_names)
    owner, faces, fuv = [], [], []
    for gi, (_, _, fl) in enumerate(groups):
        for fi in fl:
            f = base.obj.faces[fi]
            faces.append([v for v, _ in f])
            fuv.append([base.obj.vt[t] if t >= 0 else (0.0, 0.0) for _, t in f])
            owner.append(gi)
    Wd = np.zeros((base.nV, nb), np.float32)
    for k in range(4):
        np.add.at(Wd, (np.arange(base.nV), J[:, k].astype(np.int64)), W[:, k])
    P = VA.copy()
    D = {k: v.copy() for k, v in (shapes_mh or {}).items()}
    nV = base.nV
    for lev in range(levels):
        S, nf, csrc = catmull_clark(faces, nV)
        P = S @ P
        Wd = S @ Wd
        D = {k: S @ v for k, v in D.items()}
        nuv, nfaces, nowner = [], [], []
        for (fi, src), ids in zip(csrc, nf):
            uv = np.asarray(fuv[fi])
            cu = []
            for s in src:
                if s[0] == "v":
                    cu.append(uv[s[1]])
                elif s[0] == "e":
                    cu.append(0.5 * (uv[s[1]] + uv[s[2]]))
                else:
                    cu.append(uv.mean(0))
            nuv.append(cu)
            nfaces.append(list(ids))
            nowner.append(owner[fi])
        faces, fuv, owner, nV = nfaces, nuv, nowner, P.shape[0]
    Wd = np.asarray(Wd)
    top = np.argsort(-Wd, axis=1)[:, :4]
    Wt = np.take_along_axis(Wd, top, 1)
    Wt = np.clip(Wt, 0, None)
    Wt /= np.maximum(Wt.sum(1, keepdims=True), 1e-9)
    parts = []
    for gi, (name, material, _) in enumerate(groups):
        keymap, orig, uvs, tris = {}, [], [], []
        for f, fu, ow in zip(faces, fuv, owner):
            if ow != gi:
                continue
            loc = []
            for v, uv in zip(f, fu):
                k = (v, round(float(uv[0]), 6), round(float(uv[1]), 6))
                if k not in keymap:
                    keymap[k] = len(orig)
                    orig.append(v)
                    uvs.append(uv)
                loc.append(keymap[k])
            tris += tri_of(loc)
        orig = np.array(orig, np.int64)
        shapes = {}
        for sn, d in D.items():
            dl = np.asarray(d)[orig]
            sel = np.where(np.linalg.norm(dl, axis=1) > 1e-8)[0]
            if len(sel):
                shapes[sn] = (sel, dl[sel])
        parts.append(Part(name, material, P[orig], np.array(uvs), np.array(tris), top[orig].astype(np.uint16),
                          Wt[orig], shapes=shapes, weld=orig))
    return parts
