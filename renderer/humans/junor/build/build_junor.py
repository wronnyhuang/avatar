"""junor — build junor.asset (juno.asset/1 format): ICT-FaceKit scan head (MIT) grafted on an MPFB2 (CC0) adult
female body, ARKit + juno extras + correctives, juno-style eyeballs/cornea/occlusion/tearline/caruncle fitted to the
ICT lids, ICT teeth/gums/tongue/mouth socket, the juno wardrobe refit on the adult body, procedural skin maps.

usage: python build_junor.py <mpfb_data_dir> <ict_npz> <out_asset_dir> [--identity identity.npy] [--version vN]
  mpfb_data_dir = asset-src/mpfb2/data (MPFB2 src/mpfb/data), ict_npz = asset-src/ict.npz (get_sources.sh / ict_io.py from ICT-FaceKit)
Hair is the HAIR piece's (no hair groups here; tex/scalp_mask.png + landmarks hairline/brow/lash anchors provided).
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
JB = os.path.abspath(os.path.join(HERE, "..", "..", "juno", "build"))
for p in (HERE, JB, os.path.abspath(os.path.join(HERE, "..", "..", ".."))):
    if p not in sys.path:
        sys.path.insert(0, p)
from humans import asset_io  # noqa: E402
import graft  # noqa: E402
import face_r  # noqa: E402
import tex_r  # noqa: E402
from parts import Part, export, mh_part  # noqa: E402
from juno_base import vertex_normals, tri_of  # noqa: E402

DEFAULT_ID = os.path.join(HERE, "identity.npy")
IRIS_R_M = 0.0059          # real iris radius (11.8 mm diameter)
PUPIL_K = 0.36


def ict_part(H, name, mat, F, FT, J, W, S=None, **kw):
    """Part from ICT faces (quads/tris), vertices split at UV seams; weld = ICT vertex id."""
    keymap, orig, uvs, tris = {}, [], [], []
    for f, ft in zip(F, FT):
        loc = []
        for v, t in zip(f, ft):
            if v < 0:
                continue
            k = (int(v), int(t))
            if k not in keymap:
                keymap[k] = len(orig)
                orig.append(int(v))
                uvs.append(H.VT[t])
            loc.append(keymap[k])
        tris += tri_of(loc)
    orig = np.array(orig, np.int64)
    shapes = {}
    for sn, d in (S or {}).items():
        dl = d[orig]
        sel = np.where(np.linalg.norm(dl, axis=1) > 1e-8)[0]
        if len(sel):
            shapes[sn] = (sel, dl[sel])
    return Part(name, mat, H.P[orig], np.array(uvs), np.array(tris), J[orig], W[orig], shapes=shapes, weld=orig,
                orig=orig, **kw)


def faces_by_vertices(H, regions_or_ids, F, FT):
    ids = np.zeros(H.N, bool)
    ids[regions_or_ids] = True
    m = np.array([np.mean([ids[v] for v in f if v >= 0]) > 0.5 for f in F])
    return F[m], FT[m]


class Shim:
    """juno eyes_geo.EyeFit wants a base: VA, tris, skin_mask, eye."""
    def __init__(self, base, H):
        self.VA = H.P
        F, _ = H.skin_faces()
        self.tris = np.array([t for f in F for t in tri_of([v for v in f if v >= 0])], np.int64)
        self.skin_mask = H.region_mask("face", "headneck", "eye_socket_l", "eye_socket_r")
        self.eye = base.eye
        self.bidx = base.bidx


def eye_parts(base, H, S, texdir):
    import eyes_geo as eg
    import eyes_tex as et
    from eyes_mouth import _save, _flat_set
    shim = Shim(base, H)
    parts, lod1, mats, rep = [], {}, {}, {}
    bi = base.bidx
    fits = {}
    for side in ("l", "r"):
        fit = eg.EyeFit(shim, S, side)
        c = fit.c
        ir = H.P[H.region(f"iris_{side}")] - c
        ri = min(IRIS_R_M, 0.75 * fit.R)
        fit.theta_iris = float(np.arcsin(ri / fit.R))
        fit.theta_limbus = fit.theta_iris * 1.04
        tc = np.clip(fit.theta_m - eg.GAZE_DEG * eg.D2R, 0, None)
        clear = (fit.lid_clear["rest"] - 1.0 - 0.012) * fit.R
        fit.bulge = float(np.clip(clear / max(fit.bulge_profile(tc).max(), 1e-3), 0.02 * fit.R, 0.06 * fit.R))
        fits[side] = fit
        rep[side] = dict(fit.report(), ict_iris_r_mm=round(float(np.hypot(ir[:, 0], ir[:, 1]).max()) * 1000, 2))
        parts.append(eg.eyeball_part(fit, side, bi[f"eye_{side}"]))
        parts.append(eg.cornea_part(fit, side, bi[f"eye_{side}"]))
        lod1[f"eye_{side}"] = eg.eyeball_part(fit, side, bi[f"eye_{side}"], nphi=24)
        lod1[f"cornea_{side}"] = eg.cornea_part(fit, side, bi[f"eye_{side}"], nphi=24)
        parts.append(eg.occlusion_part(fit, side, bi["head"]))
        parts.append(eg.tearline_part(fit, side, bi["head"]))
        parts.append(eg.caruncle_part(fit, side, bi["head"]))
    f = fits["l"]
    iris_uv = 0.5 * f.theta_iris / np.pi
    pupil_uv = iris_uv * PUPIL_K
    if os.environ.get("JUNOR_EYE_TEX", "r") == "v6":
        alb, nrm, pk = et.eye_maps(2048, iris_uv, pupil_uv)
    else:   # HEAD-MTEX: polar-built iris (fibres, crypts, collarette, ruff, furrows) + corner-vessel sclera, 1024
        import eyes_r
        alb, nrm, pk = eyes_r.eye_maps(1024, iris_uv, pupil_uv)
    for side in ("l", "r"):
        _save(texdir, f"eye_{side}_albedo", alb if side == "l" else alb[:, ::-1])
        _save(texdir, f"eye_{side}_normal", nrm if side == "l" else nrm[:, ::-1] * [-1, 1, 1] + [255, 0, 0])
        _save(texdir, f"eye_{side}_packed", pk if side == "l" else pk[:, ::-1])
        mats[f"eye_{side}"] = {"type": "eye", "albedo": f"tex/eye_{side}_albedo.png", "normal": f"tex/eye_{side}_normal.png",
                               "packed": f"tex/eye_{side}_packed.png",
                               "params": {"iris_uv_radius": float(iris_uv), "pupil_uv_radius": float(pupil_uv),
                                          "limbus_width": 0.06, "ior": 1.376,
                                          "iris_concavity_m": float(0.07 * f.R * np.sin(f.theta_iris))}}
    mats["caruncle"] = {"type": "gums", **_flat_set(texdir, "caruncle", (226, 150, 140), 0.16, n=64, thick=0.6), "params": {}}
    mats["cornea"] = {"type": "cornea", "params": {"ior": 1.376, "roughness": 0.02, "alpha": 0.06}}
    _save(texdir, "eye_occlusion_albedo", et.strip_occlusion())
    mats["eye_occlusion"] = {"type": "eye_occlusion", "albedo": "tex/eye_occlusion_albedo.png", "params": {"alpha_from": "albedo.a"}}
    _save(texdir, "tearline_albedo", et.strip_tearline())
    mats["tearline"] = {"type": "tearline", "albedo": "tex/tearline_albedo.png",
                        "params": {"alpha_from": "albedo.a", "roughness": 0.03, "ior": 1.33}}
    return parts, lod1, mats, rep


def head_concavity_ao(part, iters=3):
    """per-vertex cavity AO from mean-curvature concavity (ears' concha, nostrils, lid creases, mouth corners)."""
    from scipy.sparse import coo_matrix
    u, inv = np.unique(part.weld, return_inverse=True)
    P = np.zeros((len(u), 3))
    P[inv] = part.P
    T = inv[part.tris]
    r = np.concatenate([T[:, 0], T[:, 1], T[:, 2], T[:, 1], T[:, 2], T[:, 0]])
    c = np.concatenate([T[:, 1], T[:, 2], T[:, 0], T[:, 0], T[:, 1], T[:, 2]])
    A = coo_matrix((np.ones(len(r)), (r, c)), shape=(len(u), len(u))).tocsr()
    A.data[:] = 1
    deg = np.asarray(A.sum(1)).ravel()
    N = vertex_normals(P, T)
    ao = np.ones(len(u))
    Q = P.copy()
    for s_ in (1, 3, 9):              # multi-scale: smooth k rings, concavity = (avg - P) . N
        for _ in range(s_):
            Q = (A @ Q) / np.maximum(deg, 1)[:, None]
        conc = ((Q - P) * N).sum(1)
        ao *= np.clip(1 - np.clip(conc, 0, None) / (0.0012 * s_), 0.7, 1)
    for _ in range(iters):
        ao = 0.5 * ao + 0.5 * (A @ ao) / np.maximum(deg, 1)
    return ao[inv]


def corner_arrays(part):
    T = part.tris
    return part.UV[T], part.P[T], part.N[T] if part.N is not None else part.welded_normals()[T]


def unify_normals(parts):
    """one smooth normal field across the welded skin parts (head + bridge + body): positions welded at 1e-6 m."""
    P = np.concatenate([p.P for p in parts])
    key = np.round(P / 1e-6).astype(np.int64)
    _, inv = np.unique(key, axis=0, return_inverse=True)
    inv = inv.reshape(-1)
    off, T = 0, []
    for p in parts:
        T.append(inv[p.tris + off])
        off += len(p.P)
    Pu = np.zeros((inv.max() + 1, 3))
    Pu[inv] = P
    n = vertex_normals(Pu, np.concatenate(T))
    off = 0
    for p in parts:
        p.N = n[inv[off:off + len(p.P)]]
        off += len(p.P)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data")
    ap.add_argument("ict")
    ap.add_argument("out")
    ap.add_argument("--identity", default=DEFAULT_ID)
    ap.add_argument("--version", default="v1")
    ap.add_argument("--wardrobe", default="wardrobe_r", help="module with build(base, texdir) (fallback: juno wardrobe)")
    ap.add_argument("--tex", type=int, default=2048)
    a = ap.parse_args()
    t0 = time.time()
    out = a.out
    if os.path.exists(out):
        shutil.rmtree(out)
    texdir = os.path.join(out, "tex")
    os.makedirs(texdir)
    rep = {}
    base = graft.make_base(a.data, a.ict, np.load(a.identity), log=lambda s: (print(s), rep.setdefault("log", []).append(s)))
    H = base.ict
    S, CORR = face_r.build(base)
    names = list(asset_io.REQUIRED_SHAPES) + sorted(k for k in S if k not in asset_io.REQUIRED_SHAPES)
    missing = [s for s in asset_io.REQUIRED_SHAPES if s not in S]
    assert not missing, missing
    J, W = base.ict_JW
    mats = {}
    # ---- ICT parts
    Fs, FTs = H.faces("face", "headneck", "eye_socket_l", "eye_socket_r")
    head = ict_part(H, "head", "skin_head", Fs, FTs, J, W, S)
    Fm, FTm = H.faces("mouth_socket")
    mouth = ict_part(H, "mouth_interior", "mouth_interior", Fm, FTm, J, W, S)
    Ft, FTt = H.faces("teeth_upper")
    teeth_u = ict_part(H, "teeth_upper", "teeth", Ft, FTt, J, W)
    Ft, FTt = H.faces("teeth_lower")
    teeth_l = ict_part(H, "teeth_lower", "teeth", Ft, FTt, J, W)
    Fg, FTg = H.faces("gums_tongue")
    gu = faces_by_vertices(H, np.setdiff1d(H.upper_gt, H.tongue_ids), Fg, FTg)
    gl = faces_by_vertices(H, H.lower_gum_ids, Fg, FTg)
    tg = faces_by_vertices(H, H.tongue_ids, Fg, FTg)
    gums_u = ict_part(H, "gums_upper", "gums", *gu, J, W)
    gums_l = ict_part(H, "gums_lower", "gums", *gl, J, W, S)
    tongue = ict_part(H, "tongue", "tongue", *tg, J, W, S)
    # ICT UVs are UDIM-like tiles (face 0-1, backhead 1-2, gums 2-3, teeth 3-4, lashes 6-7): the head skin gets a 2-tile
    # atlas (u / 2), every other part is moved into 0..1
    head.UV[:, 0] /= 2.0
    for p_ in (mouth, teeth_u, teeth_l, gums_u, gums_l, tongue):
        p_.UV[:, 0] -= np.floor(p_.UV[:, 0].min() + 1e-6)
    e_parts, e_lod1, e_mats, rep["eyes"] = eye_parts(base, H, S, texdir)
    mats.update(e_mats)
    # ICT lash cards (scan-fitted rows that ride the lid shapes); a flat albedo -> the renderer's fix_lash_cards strands
    lash = []
    for side in ("l", "r"):
        Fl, FTl = H.faces(f"lashes_{side}")
        lash.append(ict_part(H, f"lashes_{side}", "lashes", Fl, FTl, J, W, S, double_sided=True))
    from eyes_mouth import _flat_set
    mats["lashes"] = {"type": "lashes", **_flat_set(texdir, "lashes", (34, 22, 17), 0.4, n=16), "params": {"role": "ICT lash cards"}}
    # ---- wardrobe
    try:
        import importlib
        wd = importlib.import_module(a.wardrobe)
    except ImportError:
        import wardrobe as wd
    rep["wardrobe_module"] = wd.__name__
    wd_parts, wd_mats, wd_hide = wd.build(base, texdir)
    mats.update(wd_mats)
    # ---- body skin (MH faces below the weld, minus what the clothes cover) + bridge
    body_f = [fi for fi in base.faces_of("body") if fi not in base.cut_faces and fi not in wd_hide]
    body = mh_part(base, "body", "skin_body", body_f)
    br = base.bridge
    bridge = Part("neck_bridge", "skin_body", br["P"], br["UV"], br["tris"], br["J"], br["W"])
    unify_normals([head, bridge, body])
    # ---- textures
    tex_r.HEAD_AO = head_concavity_ao(head)[head.tris]
    mats.update(tex_r.skin(base, corner_arrays(head), _cat_corners([body, bridge]), texdir, n=a.tex))
    mats["skin_head"]["params"]["wrinkle"] = tex_r.wrinkles(base, corner_arrays(head), texdir)
    if os.environ.get("JUNOR_MOUTH_TEX", "r") == "v6":   # the v1-v6 flat painted sets
        mats["mouth_interior"] = tex_r.painted("mouth_interior", *corner_arrays(mouth)[:2], _mouth_fn(base), texdir,
                                               typ="mouth_interior", rough=0.4, spec=0.45, thick=0.4)
        mats["teeth"] = tex_r.painted("teeth", *_cat_corners([teeth_u, teeth_l])[:2], _teeth_fn(base), texdir, typ="teeth",
                                      rough=0.22, spec=0.5, thick=0.35)
        mats["gums"] = tex_r.painted("gums", *_cat_corners([gums_u, gums_l])[:2], _gums_fn(base),
                                     texdir, typ="gums", rough=0.32, thick=0.5, n=128)
        mats["tongue"] = tex_r.painted("tongue", *corner_arrays(tongue)[:2], _tongue_fn(base), texdir, typ="tongue",
                                       rough=0.42, thick=0.5, n=256)
    else:   # HEAD-MTEX: per-tooth / per-tissue procedural maps (mouth_r.py)
        import mouth_r
        mlm = base.lm["mouth"]
        mp = lambda p_: (p_.name, p_.P, p_.UV, p_.tris, p_.N if p_.N is not None else p_.welded_normals())
        TT, GG = [mp(teeth_u), mp(teeth_l)], [mp(gums_u), mp(gums_l)]
        mats["teeth"] = mouth_r.teeth(texdir, TT, GG, mlm, n=1024, spec=0.5, thick=0.35)
        mats["gums"] = mouth_r.gums(texdir, GG, TT, mlm, n=512, thick=0.5)
        mats["tongue"] = mouth_r.tongue(texdir, [mp(tongue)], mlm, n=512, thick=0.5)
        mats["mouth_interior"] = mouth_r.mouth_interior(texdir, [mp(mouth)], mlm, n=512, thick=0.4)
    # teeth_lower shares the teeth material: its UVs are in the same ICT teeth atlas
    parts = [head, body, bridge] + e_parts + lash + [mouth, teeth_u, teeth_l, gums_u, gums_l, tongue] + wd_parts
    if os.environ.get("JUNOR_DUMP_PARTS"):          # dev: pickle the lod0 parts for offline lod1 work
        import pickle
        with open(os.environ["JUNOR_DUMP_PARTS"], "wb") as f:
            pickle.dump({"parts": parts, "e_lod1": e_lod1, "eyes": {s: base.eye[s][:2] for s in ("l", "r")},
                         "names": names, "cmeta": {nm: {"corrective": {"of": list(dr), "rule": "product"}}
                                                   for nm, dr in CORR.items()}}, f)
    if os.environ.get("JUNOR_LOD1_DECIMATE", "1") == "1":    # head-lod: lod1 < 60k tris (0 = the v6 lod1)
        import lod_r
        parts1 = lod_r.lod1_parts(parts, e_lod1, eyes={s: base.eye[s][:2] for s in ("l", "r")},
                                  mouth=np.asarray(base.lm["mouth"]), log=lambda s: (print(s), rep.setdefault("lod1", []).append(s)))
    else:
        parts1 = [e_lod1.get(p.name, p) for p in parts]
    head1 = next(p for p in parts1 if p.name == "head")
    for p in parts:
        for sn in p.shapes:
            if sn not in names:
                names.append(sn)
    cmeta = {nm: {"corrective": {"of": list(dr), "rule": "product"}} for nm, dr in CORR.items()}
    os.makedirs(os.path.join(out, "hair"), exist_ok=True)
    export(out, "lod0", parts, names, cmeta)
    export(out, "lod1", parts1, names, cmeta)
    if os.environ.get("JUNOR_SHAPE_PRUNE", "1") == "1":   # head-lod: drop numerically-zero shape entries
        import shape_prune
        for lod in ("lod0", "lod1"):
            shape_prune.prune(os.path.join(out, lod), log=lambda s: (print(s), rep.setdefault("log", []).append(s)))
    skel = dict(base.skeleton)
    jz = base.jaw
    skel["skeleton_hints"] = {"jaw_open_deg": round(jz["open_deg"], 3), "jaw_open_axis": [1, 0, 0],
                              "jaw_side_deg": round(jz["side_deg"], 3), "jaw_forward_m": round(jz["forward_m"], 5),
                              "jaw_hinge": np.round(jz["hinge"], 5).tolist(),
                              "eye_center_l": np.round(base.eye["l"][0], 5).tolist(),
                              "eye_center_r": np.round(base.eye["r"][0], 5).tolist(),
                              "eye_radius_m": round(float(base.eye["l"][1]), 5)}
    with open(os.path.join(out, "skeleton.json"), "w") as f:
        json.dump(skel, f, indent=0)
    allP = np.concatenate([p.P for p in parts])
    skinP = head.P
    Hh = float(max(skinP[:, 1].max(), 0))

    def grasp(side):
        o = 0.5 * (base.cube(f"joint-{side}-finger-1-4") + base.cube(f"joint-{side}-finger-3-2"))
        Hm = base.rest[base.bidx[f"hand_{side}"]]
        G = Hm.copy()
        G[:3, 3] = o
        return {"bone": f"hand_{side}", "offset": (np.linalg.inv(Hm) @ G).reshape(-1).round(6).tolist()}

    L68 = base.lm68
    lm = base.lm
    landmarks = {"eye_l": np.round(lm["eye_c_l"], 5).tolist(), "eye_r": np.round(lm["eye_c_r"], 5).tolist(),
                 "mouth": np.round(lm["mouth"], 5).tolist(), "head_top": np.round(lm["head_top"], 5).tolist(),
                 "chin": np.round(lm["chin"], 5).tolist(), "nose_tip": np.round(lm["nose_tip"], 5).tolist(),
                 "grasp_l": grasp("l"), "grasp_r": grasp("r"),
                 "brow_l": np.round(L68[22:27], 5).tolist(), "brow_r": np.round(L68[17:22], 5).tolist(),
                 "lid_upper_l": np.round(L68[[42, 43, 44, 45]], 5).tolist(), "lid_upper_r": np.round(L68[[36, 37, 38, 39]], 5).tolist(),
                 "lid_lower_l": np.round(L68[[45, 46, 47, 42]], 5).tolist(), "lid_lower_r": np.round(L68[[39, 40, 41, 36]], 5).tolist(),
                 "multipie68": np.round(L68, 5).tolist()}
    import brows_r
    hair_groups = brows_r.write(out, head, head1, landmarks)
    for g in hair_groups:
        mats.setdefault("hair_" + g, {"type": "hair", "params": {k: hair_groups[g][k] for k in ("color_root", "color_tip", "melanin")}})
    if os.environ.get("JUNOR_TEX_PACK", "1") == "1":     # head-lod: jpg albedo / lossless webp maps, 512 wardrobe lod1
        import tex_pack
        tex_pack.finalize(out, mats, log=lambda s: (print(s), rep.setdefault("log", []).append(s)))
    meta = {"format": asset_io.FORMAT, "name": "junor", "version": a.version, "units": "m", "up": "+y", "forward": "+z",
            "height_m": round(Hh, 4), "scale_to_H": round(1 / Hh, 5),
            "bounds": {"min": allP.min(0).round(4).tolist(), "max": allP.max(0).round(4).tolist()},
            "rest_pose": "A", "skeleton": "skeleton.json",
            "lods": [{"name": "lod0", "mesh": "lod0/mesh.json", "tex_suffix": "", "hair": {g: v["lod"]["lod0"] for g, v in hair_groups.items()}},
                     {"name": "lod1", "mesh": "lod1/mesh.json", "tex_suffix": "_1k", "hair": {g: v["lod"]["lod1"] for g, v in hair_groups.items()}}],
            "shapes": names, "correctives": {k: v["corrective"] for k, v in cmeta.items()}, "materials": mats,
            "hair_groups": hair_groups, "landmarks": landmarks,
            "hair_hooks": {"scalp_submesh": "head", "scalp_mask": "tex/scalp_mask.png (head UV, 1 = hair-covered scalp)",
                           "brow_points": "landmarks brow_l/brow_r (5 pts each, inner->outer... ICT Multi-PIE order)",
                           "lash_lines": "landmarks lid_upper_*/lid_lower_*; eye_occl_* submesh edge = lid margin"},
            "source": {"head": "ICT-FaceKit (MIT) identity PCA + expression shapes", "body": "MPFB2 hm08 (CC0)"}}
    with open(os.path.join(out, "juno.asset.json"), "w") as f:
        json.dump(meta, f, indent=1)
    lic = os.path.join(HERE, "LICENSES.md")
    if os.path.exists(lic):
        shutil.copy(lic, os.path.join(out, "LICENSES.md"))
    with open(os.path.join(out, "build_report.json"), "w") as f:
        json.dump(rep, f, indent=1, default=str)
    A1 = asset_io.load_asset(out, "lod1")
    A = asset_io.load_asset(out, "lod0")
    print(json.dumps(asset_io.stats(A)["tris_by_submesh"]))
    print(json.dumps({"height_m": Hh, "tris0": asset_io.stats(A)["tris"], "tris1": asset_io.stats(A1)["tris"],
                      "shapes": len(names), "s": round(time.time() - t0, 1)}))


def _cat_corners(parts):
    a = [corner_arrays(p) for p in parts]
    return tuple(np.concatenate([x[k] for x in a]) for k in range(3))


def _mouth_fn(base):
    m = np.asarray(base.lm["mouth"])

    def fn(Pt):
        t = np.maximum(np.clip((m[2] - Pt[..., 2]) / 0.05, 0, 1), np.clip((np.abs(Pt[..., 0]) - 0.011) / 0.011, 0, 1))[..., None]
        front = np.array([196, 112, 110]) / 255
        deep = np.array([140, 66, 68]) / 255
        col = front * (1 - t) + deep * t
        return col, 1 - 0.55 * t[..., 0], np.full(Pt.shape[:-1], 0.4)
    return fn


def _teeth_fn(base):
    m = np.asarray(base.lm["mouth"])
    y_occ = m[1] - 0.0015

    def fn(Pt):
        t = np.clip((m[2] - Pt[..., 2]) / 0.04, 0, 1)[..., None]      # back teeth darker (less light reaches)
        e = np.abs(Pt[..., 1] - y_occ)[..., None]                       # distance from the biting edge
        enamel = np.array([238, 230, 212]) / 255
        incisal = np.array([214, 218, 222]) / 255                       # thin translucent edge, cool
        cervical = np.array([226, 208, 176]) / 255                      # warmer dentine near the gums
        k1 = np.clip(1 - e / 0.0018, 0, 1)
        k2 = np.clip((e - 0.004) / 0.004, 0, 1)
        col = enamel * (1 - k1 - k2 * (1 - k1)) + incisal * k1 + cervical * k2 * (1 - k1)
        col = col * (1 - 0.28 * t)
        return col, 1 - 0.5 * t[..., 0], np.full(Pt.shape[:-1], 0.2)
    return fn


def _gums_fn(base):
    """gum pink, darker toward the back and the sides (the buccal flares at the mouth corners read as bright pink
    triangles in v1-v4: no light reaches there)."""
    m = np.asarray(base.lm["mouth"])

    def fn(Pt):
        back = np.clip((m[2] - Pt[..., 2]) / 0.03, 0, 1)
        side = np.clip((np.abs(Pt[..., 0]) - 0.010) / 0.010, 0, 1)
        k = np.clip(1 - 0.55 * np.maximum(back, side), 0.3, 1)[..., None]
        col = np.array([190, 104, 106]) / 255 * k
        return col, k[..., 0], np.full(Pt.shape[:-1], 0.32)
    return fn


def _tongue_fn(base):
    m = np.asarray(base.lm["mouth"])

    def fn(Pt):
        t = np.clip((m[2] - Pt[..., 2]) / 0.05, 0, 1)[..., None]
        col = np.array([186, 92, 96]) / 255 * (1 - 0.35 * t)
        return col, 1 - 0.4 * t[..., 0], np.full(Pt.shape[:-1], 0.42)
    return fn


if __name__ == "__main__":
    main()
