"""asset — eyes (eyeball, cornea, occlusion, tearline) + mouth (teeth, gums, tongue, mouth interior)
(owner: eyes-mouth helper; v0 = procedural spheres + MH teeth/tongue helpers).

build(base, texdir, S) -> (parts: [Part], materials: {id: json}, hide_faces: set of MH face ids to drop)
S = face shapes on MH verts (so lid-following parts can copy the lid motion).
Tongue shapes (tongueOut, tongueTipUp, tongueDental, tongueBack) live on the tongue Part here.
"""
from __future__ import annotations

import os

import numpy as np
from PIL import Image

from juno_base import smoothstep
from parts import Part, mh_part
import eyes_geo as eg
import eyes_tex as et

REPORT = {}


def uv_sphere(center, radius, nu=48, nv=32):
    th = np.linspace(0, np.pi, nv + 1)
    ph = np.linspace(0, 2 * np.pi, nu + 1)
    T, Ph = np.meshgrid(th, ph, indexing="ij")
    P = np.stack([np.sin(T) * np.cos(Ph), np.sin(T) * np.sin(Ph), np.cos(T)], -1).reshape(-1, 3)
    ruv = (T / np.pi * 0.5).reshape(-1)
    UV = np.stack([0.5 + ruv * np.cos(Ph.reshape(-1)), 0.5 + ruv * np.sin(Ph.reshape(-1))], 1)
    tris, W = [], nu + 1
    for i in range(nv):
        for j in range(nu):
            a, b, c, d = i * W + j, i * W + j + 1, (i + 1) * W + j, (i + 1) * W + j + 1
            if i > 0:
                tris.append([a, c, b])
            if i < nv - 1:
                tris.append([b, c, d])
    # weld seam column + poles for normals
    weld = np.arange(len(P)).reshape(nv + 1, nu + 1)
    weld[:, nu] = weld[:, 0]
    weld[0, :] = weld[0, 0]
    weld[nv, :] = weld[nv, 0]
    return center + radius * P, P.copy(), UV, np.array(tris), weld.reshape(-1)


def _save(texdir, name, arr):
    im = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    im.save(os.path.join(texdir, name + ".png"))
    im.resize((max(1, im.width // 2), max(1, im.height // 2)), Image.LANCZOS).save(os.path.join(texdir, name + "_1k.png"))


def _flat_set(texdir, m, rgb, rough, n=256, thick=0.0):
    a = np.ones((n, n, 4)) * 255
    a[..., :3] = rgb
    _save(texdir, f"{m}_albedo", a)
    nn = np.zeros((64, 64, 3)); nn[...] = (128, 128, 255)
    _save(texdir, f"{m}_normal", nn)
    pk = np.zeros((64, 64, 4)); pk[...] = (255, rough * 255, 128, thick * 255)
    _save(texdir, f"{m}_packed", pk)
    return {"albedo": f"tex/{m}_albedo.png", "normal": f"tex/{m}_normal.png", "packed": f"tex/{m}_packed.png"}


LOD1 = {}   # lower-res replacements by part name for lod1 (build_juno swaps them in)


def build(base, texdir, S):
    parts, mats = [], {}
    bi = base.bidx
    # NOTE for the lead: mutates S (face shapes) in place — see eyes_geo. Correctives (corr_*) are skipped: they are
    # only meaningful added to their drivers (a blink's w^2 arc term alone points into the ball by design), and
    # conforming them alone corrupted the blink path ( asset-face r2).
    eg.conform_lids(base, {k: v for k, v in S.items() if not k.startswith("corr_")}, report=REPORT)
    fits = {}
    for side in ("l", "r"):
        fit = fits[side] = eg.EyeFit(base, S, side)
        parts.append(eg.eyeball_part(fit, side, bi[f"eye_{side}"]))
        parts.append(eg.cornea_part(fit, side, bi[f"eye_{side}"]))
        LOD1[f"eye_{side}"] = eg.eyeball_part(fit, side, bi[f"eye_{side}"], nphi=20)     # ASSET2: lod1 <= 60k tris
        LOD1[f"cornea_{side}"] = eg.cornea_part(fit, side, bi[f"eye_{side}"], nphi=20)
        parts.append(eg.occlusion_part(fit, side, bi["head"]))
        parts.append(eg.tearline_part(fit, side, bi["head"]))
        parts.append(eg.caruncle_part(fit, side, bi["head"]))     # asset-eyes4: pink wet mound, inner corner
    REPORT["eyes"] = {s: f.report() for s, f in fits.items()}
    # asset-eyes2: upper lash-line strip + wing (lashline_l/r, material lashline), rides the lid shapes
    import lashline
    ll_parts, ll_mats = lashline.build(base, S, texdir)
    parts += ll_parts
    mats.update(ll_mats)
    # eye textures: 2K lod0 (+ _1k), iris size from the left fit (both eyes share the design)
    f = fits["l"]
    iris_uv = 0.5 * f.theta_iris / np.pi
    pupil_uv = iris_uv * eg.PUPIL_K
    alb, nrm, pk = et.eye_maps(2048, iris_uv, pupil_uv)
    for side in ("l", "r"):
        a_ = alb if side == "l" else alb[:, ::-1]      # mirrored iris pattern (not identical twins)
        _save(texdir, f"eye_{side}_albedo", a_)
        _save(texdir, f"eye_{side}_normal", nrm if side == "l" else nrm[:, ::-1] * [-1, 1, 1] + [255, 0, 0])
        _save(texdir, f"eye_{side}_packed", pk if side == "l" else pk[:, ::-1])
        mats[f"eye_{side}"] = {"type": "eye", "albedo": f"tex/eye_{side}_albedo.png", "normal": f"tex/eye_{side}_normal.png",
                               "packed": f"tex/eye_{side}_packed.png",
                               "params": {"iris_uv_radius": float(iris_uv), "pupil_uv_radius": float(pupil_uv),
                                          "limbus_width": 0.06, "ior": 1.376, "iris_concavity_m": float(0.07 * f.R * np.sin(f.theta_iris))}}
    mats["caruncle"] = {"type": "gums", **_flat_set(texdir, "caruncle", (234, 160, 150), 0.16, n=64, thick=0.6),
                        "params": {"role": " asset-eyes4: caruncle (inner-corner wet mound)"}}
    mats["cornea"] = {"type": "cornea", "params": {"ior": 1.376, "roughness": 0.02, "alpha": 0.06}}
    _save(texdir, "eye_occlusion_albedo", et.strip_occlusion())
    mats["eye_occlusion"] = {"type": "eye_occlusion", "albedo": "tex/eye_occlusion_albedo.png", "params": {"alpha_from": "albedo.a"}}
    _save(texdir, "tearline_albedo", et.strip_tearline())
    mats["tearline"] = {"type": "tearline", "albedo": "tex/tearline_albedo.png",
                        "params": {"alpha_from": "albedo.a", "roughness": 0.03, "ior": 1.33}}
    # ---- mouth interior: the hm08 cavity faces (CC0 mpfb inside-mouth UV mask) cut out of the skin, own dark material
    hide = set()
    cav = _cavity_faces(base)
    hide |= set(cav.tolist())
    mi = mh_part(base, "mouth_interior", "mouth_interior", cav, shapes_mh=S)
    parts.append(mi)
    mats["mouth_interior"] = {"type": "mouth_interior", **_cavity_tex(base, mi, texdir), "params": {}}
    # ---- teeth + gums: procedural stylised arches fitted to the lip lining (mouth_teeth, asset-face r3);
    # tongue from the MH helper
    import face_shapes
    import mouth_teeth
    from face_rig import FaceRig
    rig = getattr(face_shapes.build, "rig", None) or FaceRig(base)
    cv = np.unique(np.concatenate([np.array([v for v, _ in base.obj.faces[f]]) for f in cav]))
    tparts, trep = mouth_teeth.build(base, rig, cv)
    parts += tparts
    REPORT["teeth"] = trep
    VA = base.VA.copy()
    tv = base.verts("helper-tongue")
    tz = VA[tv, 2]
    tip = np.zeros(base.nV)
    tip[tv] = smoothstep(tz.min(), tz.max(), tz)
    back = np.zeros(base.nV)
    back[tv] = 1 - tip[tv]
    TS = {"tongueOut": (tip ** 1.5)[:, None] * np.array([0, -0.006, 0.030]),
          "tongueTipUp": (tip ** 2)[:, None] * np.array([0, 0.008, 0.002]),
          "tongueDental": (tip ** 2)[:, None] * np.array([0, 0.002, 0.010]),
          "tongueBack": (back ** 2)[:, None] * np.array([0, 0.006, 0])}
    parts.append(mh_part(base, "tongue", "tongue", base.faces_of("helper-tongue"), VA=VA, shapes_mh=TS))
    mats["teeth"] = {"type": "teeth", **_flat_set(texdir, "teeth", (250, 248, 242), 0.22, thick=0.35), "params": {}}
    mats["gums"] = {"type": "gums", **_flat_set(texdir, "gums", (196, 104, 108), 0.34, thick=0.5), "params": {}}
    mats["tongue"] = {"type": "tongue", **_flat_set(texdir, "tongue", (176, 78, 82), 0.38, thick=0.5), "params": {}}
    return parts, mats, hide


def _cavity_faces(base):
    im = np.asarray(Image.open(os.path.join(base.data, "textures", "mpfb_inside-mouth.jpg")).convert("L"), float) / 255
    H, W = im.shape
    out = []
    for fi in base.faces_of("body"):
        uv = np.array([base.obj.vt[t] for _, t in base.obj.faces[fi]]).mean(0)
        if im[int((1 - uv[1]) * (H - 1)), int(uv[0] * (W - 1))] > 0.5:
            out.append(fi)
    return np.array(out, np.int64)


def _cavity_tex(base, part, texdir, n=512):
    """albedo painted in the cavity's own (MH) UVs: warm pink near the lips -> deep red-brown at the back."""
    from scipy.spatial import cKDTree
    m = base.lm["mouth"]
    depth = np.clip((m[2] - part.P[:, 2]) / 0.045, 0, 1)            # 0 at the lips .. 1 deep
    kd = cKDTree(part.UV)
    yy, xx = np.mgrid[0:n, 0:n]
    q = np.stack([(xx + 0.5) / n, 1 - (yy + 0.5) / n], -1).reshape(-1, 2)
    dd, ii = kd.query(q)
    t = depth[ii].reshape(n, n)
    front = np.array([168, 74, 70.0])
    deep = np.array([70, 22, 22.0])
    col = front * (1 - t[..., None]) + deep * t[..., None]
    a = np.concatenate([col, np.full((n, n, 1), 255.0)], -1)
    e = _flat_set(texdir, "mouth_interior", (0, 0, 0), 0.45, thick=0.4)
    _save(texdir, "mouth_interior_albedo", a)
    pk = np.zeros((n, n, 4))
    pk[..., 0] = (1 - 0.6 * t) * 255
    pk[..., 1] = 0.45 * 255
    pk[..., 2] = 0.35 * 255
    pk[..., 3] = 0.4 * 255
    _save(texdir, "mouth_interior_packed", pk)
    return e
