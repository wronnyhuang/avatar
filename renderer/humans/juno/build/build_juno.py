"""asset — build the Juno asset (juno.asset/1) from the MPFB2 CC0 data + our own procedural modules.

usage: python build_juno.py <mpfb_data_dir> <out_asset_dir> [--skip hair,wardrobe] [--subdiv 1]
  mpfb_data_dir = T/asset-src/mpfb2/data
Pipeline: juno_base.Base (design + skeleton + weights) -> face_shapes -> skin_tex -> eyes_mouth -> wardrobe -> hair
-> skin parts (head/body, minus faces hidden by clothes) -> export lod0 (+ subdivision) and lod1 -> validate.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", "..")))
from humans import asset_io  # noqa: E402
import juno_base  # noqa: E402
import face_shapes, eyes_mouth, skin_tex, hair, wardrobe  # noqa: E402,E401
from parts import export, mh_part  # noqa: E402


def skin_parts(base, S, hide, subdiv=0):
    head_bones = {base.bidx[b] for b in ("head", "neck_01", "jaw", "eye_l", "eye_r")}
    hw = np.zeros(base.nV)
    for k in range(4):
        hw += np.isin(base.J[:, k], list(head_bones)) * base.W[:, k]
    head_f, body_f = [], []
    for fi in base.faces_of("body"):
        if fi in hide:
            continue
        f = base.obj.faces[fi]
        (head_f if np.mean([hw[vi] for vi, _ in f]) > 0.5 else body_f).append(fi)
    if subdiv:
        from parts import mh_subdiv_parts
        return mh_subdiv_parts(base, [("head", "skin_head", head_f), ("body", "skin_body", body_f)], S, levels=subdiv)
    return [mh_part(base, "head", "skin_head", head_f, shapes_mh=S), mh_part(base, "body", "skin_body", body_f, shapes_mh=S)]


MOUTH_CULL = False


def cull_inner_faces(base, em_parts):
    """RENDER asks (v2): (1) head faces lying inside the eyeball in front (between iris and cornea) -> cull; (2) inner
    mouth faces that sit in front of / around the teeth+tongue (open mouth rendered black) -> cull. Returns MH face ids."""
    V = base.VA
    out = set()
    fl = base.faces_of("body")
    cen = np.array([V[[v for v, _ in base.obj.faces[fi]]].mean(0) for fi in fl])
    for side in ("l", "r"):
        c, R = base.eye[side]
        q = cen - c
        d = np.linalg.norm(q, axis=1)
        m = (d < 0.995 * R) & (q[:, 2] > 0.35 * d)
        out |= set(fl[m].tolist())
    teeth = [p for p in em_parts if p.name.startswith(("teeth", "tongue", "gums"))]
    if teeth and MOUTH_CULL:   # OFF: a bbox cull cut holes in the cheeks/lip corners (v4 sheet) -> successor: cull by
                               # cavity topology (mpfb_inside-mouth mask) + visibility from the front instead
        P = np.concatenate([p.P for p in teeth])
        lo, hi = P.min(0) - [0.002, 0.002, 0.0], P.max(0) + [0.002, 0.002, 0.0]
        inside = np.all((cen >= lo) & (cen <= hi), axis=1)
        # keep the visible lip skin: only faces behind the front-most teeth surface
        inside &= cen[:, 2] < P[:, 2].max() - 0.001
        out |= set(fl[inside].tolist())
    return out


def bind_roots(part, roots):
    """nearest triangle of a Part for each root -> (tri local index, bary (S,3), uv (S,2))."""
    from scipy.spatial import cKDTree
    P, T = part.P, part.tris
    cen = P[T].mean(1)
    _, cand = cKDTree(cen).query(roots, k=min(12, len(T)))
    best_t = np.zeros(len(roots), np.int64)
    best_b = np.zeros((len(roots), 3))
    best_d = np.full(len(roots), np.inf)
    for k in range(cand.shape[1]):
        t = cand[:, k]
        a, b, c = P[T[t, 0]], P[T[t, 1]], P[T[t, 2]]
        v0, v1, v2 = b - a, c - a, roots - a
        d00, d01, d11 = (v0 * v0).sum(1), (v0 * v1).sum(1), (v1 * v1).sum(1)
        d20, d21 = (v2 * v0).sum(1), (v2 * v1).sum(1)
        den = np.maximum(d00 * d11 - d01 * d01, 1e-18)
        v = (d11 * d20 - d01 * d21) / den
        w = (d00 * d21 - d01 * d20) / den
        bc = np.clip(np.stack([1 - v - w, v, w], 1), 0, 1)
        bc /= bc.sum(1, keepdims=True)
        q = bc[:, 0:1] * a + bc[:, 1:2] * b + bc[:, 2:3] * c
        d = np.linalg.norm(q - roots, axis=1)
        m = d < best_d
        best_d[m], best_t[m], best_b[m] = d[m], t[m], bc[m]
    uv = (best_b[:, :, None] * part.UV[T[best_t]]).sum(1)
    return best_t, best_b, uv, best_d


def write_hair(out, parts_by_lod, hair_raw):
    groups = {}
    for g, h in hair_raw.items():
        meta = dict(h["meta"])
        sub = meta.setdefault("surface_submesh", "head")
        meta.setdefault("bone", "head")
        meta.setdefault("follow", "bone")
        meta["lod"], meta["lod_strands"] = {}, {}
        for lod in ("lod0", "lod1"):
            arr = h[lod]
            pos = np.asarray(arr["pos"], np.float32)
            S_, P_ = pos.shape[:2]
            byname = {p.name: p for p in parts_by_lod[lod]}
            tri, bary, uv, dist = bind_roots(byname[sub], pos[:, 0].astype(np.float64))
            fn = f"hair/{lod}_{g}.bin"
            asset_io.write_hair(os.path.join(out, fn), pos, uv, arr["width"], arr["seed"], arr["clump"],
                                tri=tri, bary=bary)
            meta["lod"][lod] = fn
            meta["lod_strands"][lod] = int(S_)
            meta["points"] = int(P_)
            if lod == "lod0":
                meta["strands"] = int(S_)
                meta["root_dist_max_mm"] = round(float(dist.max()) * 1000, 2)
        groups[g] = meta
    return groups


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data")
    ap.add_argument("out")
    ap.add_argument("--skip", default="")
    ap.add_argument("--version", default="v1")
    ap.add_argument("--subdiv", type=int, default=1, help="Catmull-Clark levels on the skin for lod0")
    a = ap.parse_args()
    skip = set(filter(None, a.skip.split(",")))
    t0 = time.time()
    out = a.out
    if os.path.exists(out):
        shutil.rmtree(out)
    texdir = os.path.join(out, "tex")
    os.makedirs(texdir)
    os.makedirs(os.path.join(out, "hair"))
    base = juno_base.Base(a.data)
    S = face_shapes.build(base)
    missing = [s for s in asset_io.REQUIRED_SHAPES if s not in S]
    assert not missing, missing
    names = list(asset_io.REQUIRED_SHAPES) + sorted(k for k in S if k not in asset_io.REQUIRED_SHAPES)
    mats = {}
    mats.update(skin_tex.build(base, texdir))
    em_parts, em_mats, em_hide = eyes_mouth.build(base, texdir, S)
    mats.update(em_mats)
    if "wardrobe" in skip:
        wd_parts, wd_mats, wd_hide = [], {}, set()
    else:
        wd_parts, wd_mats, wd_hide = wardrobe.build(base, texdir)
    mats.update(wd_mats)
    hair_raw = {} if "hair" in skip else hair.build(base, S)
    hide = set(em_hide) | set(wd_hide) | cull_inner_faces(base, em_parts)
    parts = skin_parts(base, S, hide, subdiv=a.subdiv) + em_parts + wd_parts      # lod0
    parts1 = skin_parts(base, S, hide, subdiv=0) + [eyes_mouth.LOD1.get(p.name, p) for p in em_parts] + wd_parts   # lod1
    hair_groups = write_hair(out, {"lod0": parts, "lod1": parts1}, hair_raw)
    for g, gi in hair_groups.items():
        mats.setdefault("hair_" + g, {"type": "hair", "params": {k: gi[k] for k in gi if k in ("color_root", "color_tip", "melanin")}})
    for p in parts:
        for sn in p.shapes:
            if sn not in names:
                names.append(sn)
    cmeta = {nm: {"corrective": {"of": list(dr), "rule": "product"}}
             for nm, dr in getattr(face_shapes, "CORRECTIVES", {}).items()}
    subs = export(out, "lod0", parts, names, cmeta)
    export(out, "lod1", parts1, names, cmeta)
    jaw_deg = face_shapes.jaw_open_deg(base, S)
    skel = dict(base.skeleton)
    skel["skeleton_hints"] = {"jaw_open_deg": round(jaw_deg, 2), "jaw_open_axis": [1, 0, 0], "jaw_side_deg": 4.0,
                              "jaw_forward_m": 0.003, "eye_center_l": base.lm["eye_c_l"].round(5).tolist(),
                              "eye_center_r": base.lm["eye_c_r"].round(5).tolist(),
                              "eye_radius_m": round(float(base.lm["eye_r_l"]), 5)}
    with open(os.path.join(out, "skeleton.json"), "w") as f:
        json.dump(skel, f, indent=0)
    H = base.height()

    def grasp(side):
        o = 0.5 * (base.cube(f"joint-{side}-finger-1-4") + base.cube(f"joint-{side}-finger-3-2"))
        Hm = base.rest[base.bidx[f"hand_{side}"]]
        G = Hm.copy()
        G[:3, 3] = o
        return {"bone": f"hand_{side}", "offset": (np.linalg.inv(Hm) @ G).reshape(-1).round(6).tolist()}

    allP = np.concatenate([p.P for p in parts])
    meta = {"format": asset_io.FORMAT, "name": "juno", "version": a.version, "units": "m", "up": "+y", "forward": "+z",
            "height_m": round(H, 4), "scale_to_H": round(1 / H, 5),
            "bounds": {"min": allP.min(0).round(4).tolist(), "max": allP.max(0).round(4).tolist()},
            "rest_pose": "A", "skeleton": "skeleton.json",
            "lods": [{"name": "lod0", "mesh": "lod0/mesh.json", "tex_suffix": "",
                      "hair": {g: gi["lod"]["lod0"] for g, gi in hair_groups.items()}},
                     {"name": "lod1", "mesh": "lod1/mesh.json", "tex_suffix": "_1k",
                      "hair": {g: gi["lod"]["lod1"] for g, gi in hair_groups.items()}}],
            "shapes": names, "correctives": {k: v["corrective"] for k, v in cmeta.items()}, "materials": mats, "hair_groups": hair_groups,
            "landmarks": {"eye_l": base.lm["eye_c_l"].round(5).tolist(), "eye_r": base.lm["eye_c_r"].round(5).tolist(),
                          "mouth": base.lm["mouth"].round(5).tolist(), "head_top": base.lm["head_top"].round(5).tolist(),
                          "chin": base.lm["chin"].round(5).tolist(), "nose_tip": base.lm["nose_tip"].round(5).tolist(),
                          "grasp_l": grasp("l"), "grasp_r": grasp("r")}}
    with open(os.path.join(out, "juno.asset.json"), "w") as f:
        json.dump(meta, f, indent=1)
    lic = os.path.join(HERE, "LICENSES.md")
    if os.path.exists(lic):
        shutil.copy(lic, os.path.join(out, "LICENSES.md"))
    A1 = asset_io.load_asset(out, "lod1")
    print("lod1", json.dumps(asset_io.stats(A1)["tris"]))
    A = asset_io.load_asset(out, "lod0")
    print(json.dumps(asset_io.stats(A)["tris_by_submesh"]))
    print(json.dumps({"height_m": H, "jaw_open_deg": jaw_deg, "tris": asset_io.stats(A)["tris"], "s": round(time.time() - t0, 1)}))


if __name__ == "__main__":
    main()
