"""junor — procedural texture sets (v1).  Skin colour is ONE function of the rest 3D position (landmark driven),
evaluated in the head's ICT UVs and in the body's MH UVs, so the neck weld has no colour step.

skin(base, head_uvt, head_P, head_N, body_uvt, body_P, body_N, texdir, n) -> {skin_head, skin_body} materials
  *_uvt (nT,3,2) per-corner UVs, *_P / *_N (nT,3,3) per-corner rest positions / normals.
  skin_head_albedo / _normal / _packed (ICT UV), skin_body_* (MH UV); packed = R AO, G roughness, B spec (.5 =
  default F0), A SSS thickness (juno.asset/1 convention). Also tex/scalp_mask.png (head UV: 1 = hair-covered scalp,
  soft hairline) and tex/lip_mask.png for HAIR / SHADE.
mouth/teeth/gums/tongue(...) -> flat-ish sets painted from position.
Detail normal: UV-space pore noise, scaled by the measured metres-per-UV so pores are ~0.25 mm in the world.
An optional scan detail kit (scratch skin helper: detail_normal / cavity / albedo_detail in ICT UV) is multiplied in
when DETAIL_DIR is set.
"""
from __future__ import annotations

import os

import numpy as np
from PIL import Image
from scipy import ndimage

from bake import dilate, raster

HEAD_AO = None      # (nT,3) per-corner concavity AO of the head part (build_junor.head_concavity_ao)
DETAIL_DIR = os.environ.get("JUNOR_SKIN_DETAIL", "skin_detail")
DETAIL_NORMAL_K = 0.7     # scan pore normal share (SHADE adds its own tiling micro-normal on top)
ALBEDO_DETAIL_K = 0.6
FRECKLE_K = 0.32
# v7 (HEAD2): concept-photo albedo projected onto the head atlas (scratch/head2/fit/proj_bake.py + finalize.py):
# photo_albedo.png (sRGB atlas), photo_w2.png (coverage), photo_fill.png (low-freq colour everywhere), photo_meta.json
PHOTO_DIR = os.environ.get("JUNOR_PHOTO", "photo_albedo")
PHOTO_ALBEDO_DETAIL_K = 0.12   # the photo carries its own pores: the LPS albedo ratio only lightly on top
FOREHEAD_SCAN_CUT = 0.75
EYE_SCAN_CUT = 0.7   # v11: fade the scan normal/cavity around the eyes (an older man's under-eye bags)

C_BASE = np.array([146, 103, 88.0]) / 255      # warm brown
C_FLUSH = np.array([156, 94, 82.0]) / 255
C_LIP = np.array([132, 74, 72.0]) / 255
C_UNDER = np.array([124, 86, 78.0]) / 255
C_FRECK = np.array([110, 66, 46.0]) / 255
C_SCALP = np.array([92, 64, 50.0]) / 255


def _srgb_save(path, arr, half=True):
    im = Image.fromarray(np.clip(arr * 255 + 0.5, 0, 255).astype(np.uint8))
    im.save(path)
    if half:
        im.resize((im.width // 2, im.height // 2), Image.LANCZOS).save(path.replace(".png", "_1k.png"))


def _g(P, c, s):
    return np.exp(-((P - c) ** 2).sum(-1) / (2 * s * s))


def _smooth(a, b, x):
    t = np.clip((x - a) / (b - a), 0, 1)
    return t * t * (3 - 2 * t)


class Fields:
    def __init__(self, base):
        lm = base.lm
        self.el, self.er = np.asarray(lm["eye_c_l"]), np.asarray(lm["eye_c_r"])
        self.mouth = np.asarray(lm["mouth"])
        self.nose = np.asarray(lm["nose_tip"])
        self.chin = np.asarray(lm["chin"])
        self.top = np.asarray(lm["head_top"])
        L = base.lm68
        self.lip_outer = L[48:60]
        self.brow_y = float(L[17:27, 1].mean())
        self.R = float(base.eye["l"][1])
        self.base = base
        # lip polygon in the frontal plane (x, y), 0.15 mm grid
        lo = self.lip_outer[:, :2]
        self.lip_o = lo.min(0) - 0.004
        self.lip_s = 0.00015
        sz = np.ceil((lo.max(0) + 0.004 - self.lip_o) / self.lip_s).astype(int)
        import cv2
        img = np.zeros((sz[1], sz[0]), np.uint8)
        pts = np.round((lo - self.lip_o) / self.lip_s).astype(np.int32)
        cv2.fillPoly(img, [pts], 255)
        self.lip_img = ndimage.gaussian_filter(img / 255.0, 4.0)
        self.lip_z = float(self.lip_outer[:, 2].min()) - 0.012

    def lip(self, P):
        q = np.round((P[..., :2] - self.lip_o) / self.lip_s).astype(int)
        h, w = self.lip_img.shape
        ok = (q[..., 0] >= 0) & (q[..., 0] < w) & (q[..., 1] >= 0) & (q[..., 1] < h) & (P[..., 2] > self.lip_z)
        out = np.zeros(P.shape[:-1])
        out[ok] = self.lip_img[q[ok][:, 1], q[ok][:, 0]]
        return out

    def scalp(self, P):
        """1 on the hair-covered scalp: above a hairline ~6.5 cm over the brows at the front, sweeping down behind
        the ears to the nape."""
        y = P[..., 1]
        z = P[..., 2]
        fz = self.el[2]
        front = _smooth(self.brow_y + 0.045, self.brow_y + 0.075, y)
        temple = _smooth(fz - 0.035, fz - 0.06, z) * _smooth(self.el[1] - 0.005, self.el[1] + 0.02, y)
        back = _smooth(fz - 0.09, fz - 0.11, z) * _smooth(self.mouth[1] - 0.02, self.mouth[1], y)
        return np.clip(np.maximum(np.maximum(front, temple), back), 0, 1)

    def colour(self, P):
        c = np.broadcast_to(C_BASE, P.shape).copy()
        lip = self.lip(P)[..., None]
        flush = (0.75 * (_g(P, self.el + [0.014, -0.032, 0.008], 0.017) + _g(P, self.er + [-0.014, -0.032, 0.008], 0.017))
                 + 0.65 * _g(P, self.nose, 0.011) + 0.3 * _g(P, self.chin + [0, 0.01, 0], 0.012))[..., None]
        ears = _smooth(0.062, 0.072, np.abs(P[..., 0]))[..., None] * _smooth(self.mouth[1] - 0.01, self.mouth[1] + 0.01, P[..., 1])[..., None]
        flush = np.clip(flush + 0.6 * ears, 0, 1)
        under = (_g(P, self.el + [0.0, -0.016, 0.006], 0.008) + _g(P, self.er + [0.0, -0.016, 0.006], 0.008))[..., None]
        c = c * (1 - 0.6 * flush) + C_FLUSH * 0.6 * flush
        c = c * (1 - 0.35 * under) + C_UNDER * 0.35 * under
        # soft socket shading on the upper lid / under the brow (reads as lid depth + a crease)
        sock = (_g(P, self.el + [0.002, 0.011, -0.002], 0.008) + _g(P, self.er + [-0.002, 0.011, -0.002], 0.008))[..., None]
        c = c * (1 - 0.13 * np.clip(sock, 0, 1)) * np.array([1.0, 0.97, 0.98]) ** np.clip(sock, 0, 1)
        c = c * (1 - lip) + C_LIP * lip
        sc = self.scalp(P)[..., None]
        c = c * (1 - 0.3 * sc) + C_SCALP * 0.3 * sc
        # paler forehead / temples, a touch of colour on the ears' rims and the chin
        fh = _smooth(self.brow_y + 0.005, self.brow_y + 0.03, P[..., 1])[..., None] * (1 - sc)
        c = c * (1 + 0.06 * fh)
        return c, lip[..., 0], flush[..., 0], sc[..., 0]

    def freckle_w(self, P):
        return np.clip(0.9 * _g(P, self.nose + [0, 0.012, -0.01], 0.022) + 0.8 * _g(P, self.el + [0.004, -0.028, 0.0], 0.02)
                       + 0.8 * _g(P, self.er + [-0.004, -0.028, 0.0], 0.02), 0, 1)


def _pore_height(shape, texel_m, rng):
    """pores ~0.25 mm wide on a fine bumpy field; units = metres of height."""
    pore_px = max(0.25e-3 / texel_m, 0.8)
    h = ndimage.gaussian_filter(rng.standard_normal(shape), pore_px * 0.6)
    dots = (rng.random(shape) < 0.012 / max(pore_px, 1) ** 2).astype(float)
    dots = ndimage.gaussian_filter(dots, pore_px * 0.5)
    dots /= max(dots.max(), 1e-9)
    fine = ndimage.gaussian_filter(rng.standard_normal(shape), max(pore_px * 0.3, 0.5))
    mid = ndimage.gaussian_filter(rng.standard_normal(shape), pore_px * 6)
    h = 0.6 * h / h.std() + 0.4 * fine / fine.std() + 0.5 * mid / mid.std() - 2.0 * dots
    return h * 12e-6


def _normal_from_height(h, texel_m, strength=1.0):
    gy, gx = np.gradient(h)
    nx = -gx / texel_m * strength
    ny = gy / texel_m * strength            # rows go down = -v
    nz = np.ones_like(nx)
    l = np.sqrt(nx * nx + ny * ny + nz * nz)
    return np.stack([nx / l, ny / l, nz / l], -1)


def _texel_m(uvt, P, n, u_scale=1.0):
    a3 = np.linalg.norm(np.cross(P[:, 1] - P[:, 0], P[:, 2] - P[:, 0]), axis=1) * 0.5
    d1, d2 = uvt[:, 1] - uvt[:, 0], uvt[:, 2] - uvt[:, 0]
    a2 = np.abs(d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0]) * 0.5 * u_scale
    ok = a2 > 1e-12
    return float(np.sqrt(np.median(a3[ok] / a2[ok]))) / n


def _load_detail(name, n, tiles, mode="RGB"):
    """scan detail kit (skin helper, ICT UDIM tiles: <name>.png = tile 0, <name>_back.png = tile 1) -> (n, n*tiles)."""
    out = []
    for t in range(tiles):
        p = os.path.join(DETAIL_DIR, f"{name}{'' if t == 0 else '_back'}.png")
        if not os.path.exists(p):
            return None
        out.append(np.asarray(Image.open(p).convert(mode).resize((n, n), Image.LANCZOS), float) / 255)
    return np.concatenate(out, 1)


def _photo(W, Hh):
    if not PHOTO_DIR or not os.path.exists(os.path.join(PHOTO_DIR, "photo_albedo.png")):
        return None
    ld = lambda f, mode="RGB": np.asarray(Image.open(os.path.join(PHOTO_DIR, f)).convert(mode).resize((W, Hh), Image.LANCZOS), float) / 255
    return ld("photo_albedo.png"), ld("photo_w2.png", "L"), ld("photo_fill.png")


def photo_base():
    """body skin base colour matched to the photo neck (no colour step at the weld)."""
    import json
    p = os.path.join(PHOTO_DIR or "", "photo_meta.json")
    if PHOTO_DIR and os.path.exists(p):
        return np.array(json.load(open(p))["c_base_srgb"])
    return None


def _skin_set(F, uvt, P, N, name, texdir, n, seed, face=True, tiles=1):
    """tiles = 2: the ICT head atlas (face tile + backhead tile side by side, uv u already /2) -> an (n, 2n) image."""
    rng = np.random.default_rng(seed)
    W, Hh = n * tiles, n
    R = raster(uvt, W, Hh)
    Pt = R.interp(P)
    m = R.mask
    sh = (Hh, W)
    col, lip, flush, sc = F.colour(Pt)
    tm = _texel_m(uvt, P, n, u_scale=float(tiles))
    ph = _photo(W, Hh) if face else None
    pw = np.zeros(sh)
    if ph is not None:
        palb, pw, pfill = ph
        pw = pw * m
        # unseen texels: the procedural pattern re-coloured to the photo's local low-frequency colour
        lp = ndimage.gaussian_filter(col, (24, 24, 0))
        col_u = col * np.clip(pfill / np.maximum(lp, 1e-3), 0.5, 2.0)
        col = palb * pw[..., None] + col_u * (1 - pw[..., None])
    if face:
        fw = F.freckle_w(Pt) * m * (1 - pw)
        fr_px = max(0.7e-3 / tm, 1.2)
        spots = (rng.random(sh) < 0.18 / fr_px ** 2) * fw * (0.4 + 0.6 * rng.random(sh))
        spots = ndimage.gaussian_filter(spots.astype(float), fr_px * 0.45)
        spots = np.clip(spots / max(np.percentile(spots[spots > 0], 99) if (spots > 0).any() else 1, 1e-6), 0, 1) * (1 - lip)
        col = col * (1 - FRECKLE_K * spots[..., None]) + C_FRECK * FRECKLE_K * spots[..., None]
    mot = ndimage.gaussian_filter(rng.standard_normal(sh), max(3e-3 / tm, 2))
    mot = mot / max(mot.std(), 1e-9)
    col = col * (1 + 0.035 * (1 - pw[..., None]) * mot[..., None] * np.array([1.0, 0.8, 0.7]))
    h = _pore_height(sh, tm, rng) * (1 - 0.7 * lip) * (1 - 0.5 * sc)
    if face:   # vertical lip lines
        ll = ndimage.gaussian_filter(rng.standard_normal(sh), (max(0.4e-3 / tm, 0.7), max(0.08e-3 / tm, 0.4)))
        h = h + lip * ll / max(ll.std(), 1e-9) * 15e-6
    nail = None
    if not face:   # hands: nails (raised glossy plate, lighter pink), knuckle darkening, finger creases (juno hand fields)
        import skin_tex as _jst
        HF = _jst.hand_fields(F.base, Pt)
        nail = HF["nail"] * m
        col = col * (1 - nail[..., None]) + np.array([214, 168, 156]) / 255 * nail[..., None]
        col = col * (1 - 0.10 * HF["knuck"][..., None] * np.array([1.0, 1.06, 1.08]))
        col = col * (1 + 0.05 * HF["tip"][..., None] * np.array([1.0, 0.92, 0.9]))
        h = h * (1 - nail) + 0.00010 * _smooth(0.0005, -0.0003, HF["nail_d"]) * (HF["nail_d"] < 1) - 25e-6 * HF["crease"]
    nrm = _normal_from_height(h, tm)
    cav = np.clip(0.5 + (h - ndimage.gaussian_filter(h, 3)) / 40e-6, 0, 1)
    rough_var = np.zeros(sh)
    used = []
    if DETAIL_DIR and face:
        dn = _load_detail("detail_normal", n, tiles)
        cov = _load_detail("coverage", n, tiles, "L")
        if dn is not None:
            cw = (cov if cov is not None else np.ones(sh))[..., None]
            dn = dn * 2 - 1
            # scan pores replace the procedural ones where the scan covers; lips keep the procedural lip lines
            # v8: the LPS scan is an older man's face - its forehead lines read as wrinkles at rest on her: fade the scan
            # normal/cavity on the forehead (procedural pores stay)
            fhd = _smooth(F.brow_y + 0.008, F.brow_y + 0.025, Pt[..., 1]) * (1 - sc) * FOREHEAD_SCAN_CUT
            # v11: the scan's under-eye bags / crow's-feet streaks read as a pale crinkled patch under the key: fade too
            eyd = np.minimum(np.linalg.norm(Pt - F.el, axis=-1), np.linalg.norm(Pt - F.er, axis=-1))
            eyb = _smooth(0.006, 0.011, eyd) * (1 - _smooth(0.026, 0.036, eyd)) * EYE_SCAN_CUT
            fhd = np.maximum(fhd, eyb)
            cw = cw * (1 - fhd[..., None]) if cw.ndim == 3 else cw * (1 - fhd)
            k = cw * (1 - 0.92 * lip[..., None]) * DETAIL_NORMAL_K
            nxy = nrm[..., :2] * (1 - k) + dn[..., :2] * k
            nrm = np.concatenate([nxy, np.sqrt(np.clip(1 - (nxy ** 2).sum(-1, keepdims=True), 0.05, 1))], -1)
            used.append("detail_normal")
            ad = _load_detail("albedo_detail", n, tiles)
            if ad is not None:
                kk = ((PHOTO_ALBEDO_DETAIL_K if ph is not None else ALBEDO_DETAIL_K) * cw * (1 - 0.85 * lip[..., None]))   # v11: the LPS beard-zone stubble showed through on the low-coverage jaw
                col = col * np.clip(np.clip(ad / 0.5, 0.3, 3.0) ** (kk / 2.2), 0.6, 1.4)   # linear ratio -> sRGB
                used.append("albedo_detail")
            cv = _load_detail("cavity", n, tiles, "L")
            if cv is not None:
                cwl = cw[..., 0] * (1 - 0.85 * lip)
                cav = cav * (1 - cwl) + cv * cwl
                used.append("cavity")
            rv = _load_detail("roughness_var", n, tiles, "L")
            if rv is not None:
                rough_var = (rv - 0.5) * cw[..., 0]
                used.append("roughness_var")
    ck = 0.08 if ph is not None else 0.16   # v11: lighter cavity darkening over the photo (the photo has its own)
    col = col * (1 - ck / 2 + ck * cav[..., None])
    rough = 0.47 - 0.10 * _g(Pt, F.nose + [0, 0.02, -0.01], 0.02) - 0.06 * _g(Pt, F.mouth + [0, 0.07, -0.02], 0.025) \
        - 0.12 * lip + 0.06 * (0.5 - cav) + 0.3 * rough_var
    if nail is not None:
        rough = rough * (1 - nail) + 0.28 * nail
    if face:   # wet inner lip band along the seam, drier matte vermilion border
        wet = lip * np.exp(-((Pt[..., 1] - F.mouth[1]) / 0.0022) ** 2)
        rough = rough - 0.14 * wet + 0.06 * lip * (1 - wet)
    spec = 0.5 + 0.1 * lip - 0.3 * (0.5 - cav)
    thick = 0.25 + 0.35 * _smooth(0.062, 0.075, np.abs(Pt[..., 0])) * (Pt[..., 2] < F.el[2] - 0.05) \
        + 0.3 * _g(Pt, F.nose, 0.008) + 0.25 * lip
    ao = np.clip(0.75 + 0.5 * cav, 0, 1)
    if face and HEAD_AO is not None and len(HEAD_AO) == len(uvt):
        gao = R.interp(HEAD_AO[..., None])[..., 0]
        gao = np.where(m, gao, 1.0)
        ao = ao * gao
        col = col * (0.85 + 0.15 * gao[..., None])
    alb = dilate(np.clip(col, 0, 1), m)
    nmap = dilate(nrm * 0.5 + 0.5, m)
    packed = dilate(np.stack([ao, np.clip(rough, 0.1, 1), np.clip(spec, 0, 1), np.clip(thick, 0, 1)], -1), m)
    out = {}
    for key, img in (("albedo", alb), ("normal", nmap), ("packed", packed)):
        p = os.path.join(texdir, f"{name}_{key}.png")
        _srgb_save(p, img)
        out[key] = f"tex/{name}_{key}.png"
    if face:
        _srgb_save(os.path.join(texdir, "scalp_mask.png"), dilate(sc, m)[..., None].repeat(3, -1), half=False)
        _srgb_save(os.path.join(texdir, "lip_mask.png"), dilate(lip, m)[..., None].repeat(3, -1), half=False)
    out["_used"] = used
    return out


def skin(base, head, body, texdir, n=2048):
    global C_BASE, C_FLUSH, C_UNDER, C_SCALP
    pb = photo_base()
    if pb is not None:   # keep the procedural palette's relations, re-based on the photo's neck tone
        k = pb / C_BASE
        C_BASE, C_FLUSH, C_UNDER, C_SCALP = pb, C_FLUSH * k, C_UNDER * k, C_SCALP * k
    F = Fields(base)
    th = _skin_set(F, *head, "skin_head", texdir, n, 11, face=True, tiles=2)
    tb = _skin_set(F, *body, "skin_body", texdir, n, 12, face=False)
    used = th.pop("_used"), tb.pop("_used")
    p = {"sss_color": [0.88, 0.4, 0.3], "sss_radius_mm": [3.8, 1.5, 0.75], "spec_f0": 0.028, "normal_strength": 1.0}
    return {"skin_head": {"type": "skin", **th, "params": dict(p, scalp_mask="tex/scalp_mask.png", lip_mask="tex/lip_mask.png",
                                                               atlas="ICT UV: face tile u 0..0.5, backhead tile u 0.5..1 (ICT u / 2)",
                                                               scan_detail=used[0])},
            "skin_body": {"type": "skin", **tb, "params": dict(p, sss_color=[0.86, 0.42, 0.31], sss_radius_mm=[3.4, 1.35, 0.7])}}


def painted(name, uvt, P, fn, texdir, n=512, rough=0.4, spec=0.5, thick=0.5, typ=None, params=None):
    """a material painted from position: fn(Pt (n,n,3)) -> rgb 0..1 (+ optional (ao, rough))."""
    R = raster(uvt, n, n)
    Pt = R.interp(P)
    r = fn(Pt)
    col, ao, rg = (r if isinstance(r, tuple) else (r, np.ones((n, n)), np.full((n, n), rough)))
    m = R.mask
    alb = dilate(np.concatenate([np.clip(col, 0, 1), np.ones((n, n, 1))], -1), m)
    pk = dilate(np.stack([ao, rg, np.full((n, n), spec), np.full((n, n), thick)], -1), m)
    nrm = np.zeros((64, 64, 3))
    nrm[...] = (0.5, 0.5, 1.0)
    for key, img in (("albedo", alb), ("packed", pk), ("normal", nrm)):
        _srgb_save(os.path.join(texdir, f"{name}_{key}.png"), img)
    return {"type": typ or name, "albedo": f"tex/{name}_albedo.png", "normal": f"tex/{name}_normal.png",
            "packed": f"tex/{name}_packed.png", "params": params or {}}


WRINKLE_DEPTH_M = 450e-6   # crease depth (SHADE: forehead lines ~0.3-0.6 mm; additive UDN blend)

WRINKLE_DRIVE = {
    "r": {"browInnerUp": 1.0, "browOuterUpLeft": 0.5, "browOuterUpRight": 0.5},
    "g": {"browDownLeft": 0.7, "browDownRight": 0.7, "noseSneerLeft": 0.3, "noseSneerRight": 0.3},
    "b": {"eyeSquintLeft": 0.5, "eyeSquintRight": 0.5, "cheekSquintLeft": 0.5, "cheekSquintRight": 0.5,
          "eyeHappyLeft": 0.6, "eyeHappyRight": 0.6, "mouthSmileLeft": 0.3, "mouthSmileRight": 0.3},
    "a": {"mouthSmileLeft": 0.7, "mouthSmileRight": 0.7, "cheekSquintLeft": 0.3, "cheekSquintRight": 0.3},
}


def _crease(d, w):
    """a soft V crease profile: 0 far, -1 at d = 0 (d, w in metres)."""
    return -np.exp(-(d / w) ** 2)


def wrinkles(base, head, texdir, n=1024, tiles=2):
    """expression wrinkle set (compressed-state tangent normal + 4 region masks) for SHADE (params.wrinkle):
    R forehead lines (brow raise), G glabella furrows (brow down), B crow's feet (squint / smile), A nasolabial fold
    (smile). Height fields are functions of the rest 3D position around the ICT Multi-PIE landmarks."""
    uvt, P, _ = head
    W, Hh = n * tiles, n
    R = raster(uvt, W, Hh)
    Pt = R.interp(P)
    m = R.mask
    L = base.lm68
    x, y, z = Pt[..., 0], Pt[..., 1], Pt[..., 2]
    fz = L[27, 2]
    front = _smooth(fz - 0.06, fz - 0.035, z)
    brow_y = float(L[17:27, 1].mean())
    tm = _texel_m(uvt, P, n, u_scale=float(tiles))
    # R forehead: 4 gently curved horizontal creases
    fy = y - brow_y - 0.012 + 0.9 * x * x
    hR = sum(_crease(fy - k * 0.0085, 0.0011) for k in range(4))
    mR = _smooth(-0.004, 0.003, fy) * _smooth(0.038, 0.030, fy) * _smooth(0.058, 0.04, np.abs(x)) * front
    # G glabella: two near-vertical furrows + one short horizontal at the nasal root
    gx = np.abs(x) - 0.0045 - 0.1 * (y - brow_y)
    hG = _crease(gx, 0.0009) + 0.6 * _crease(y - (L[27, 1] + 0.004), 0.0009) * _smooth(0.009, 0.004, np.abs(x))
    mG = _smooth(0.016, 0.009, np.abs(x)) * _smooth(-0.012, -0.004, y - brow_y) * _smooth(0.016, 0.006, y - brow_y) * front
    # B crow's feet: 4 rays fanning out from each outer canthus
    hB = np.zeros_like(x)
    mB = np.zeros_like(x)
    for ci, sx in ((45, 1.0), (36, -1.0)):
        c = L[ci]
        dx, dy = sx * (x - c[0]), y - c[1]
        r = np.hypot(dx, dy)
        ang = np.arctan2(dy, np.maximum(dx, 1e-6))
        rays = sum(_crease((ang - a) * r, 0.0007) for a in np.radians([-38, -14, 10, 32]))
        reg = _smooth(0.003, 0.007, r) * _smooth(0.024, 0.016, r) * (dx > 0) * _smooth(-0.06, -0.03, z - c[2])
        hB += rays * reg
        mB = np.maximum(mB, reg)
    # A nasolabial: groove from above the nose wing to beyond the mouth corner
    hA = np.zeros_like(x)
    mA = np.zeros_like(x)
    for wi, mi, sx in ((35, 54, 1.0), (31, 48, -1.0)):
        a = L[wi] + np.array([sx * 0.006, 0.004, -0.004])
        b = L[mi] + np.array([sx * 0.009, -0.010, -0.006])
        ab = b - a
        q = Pt - a
        t = np.clip((q @ ab) / (ab @ ab), 0, 1)
        d = np.linalg.norm(q - t[..., None] * ab, axis=-1)
        hA += 3.0 * _crease(d, 0.0024) * _smooth(0.0, 0.15, t) * _smooth(1.0, 0.8, t)
        mA = np.maximum(mA, _smooth(0.012, 0.005, d) * _smooth(-0.05, 0.1, t) * _smooth(1.1, 0.9, t))
    h = (hR * mR + hG * mG + hB * mB + hA * mA) * WRINKLE_DEPTH_M
    nrm = _normal_from_height(ndimage.gaussian_filter(h, 0.7), tm)
    _srgb_save(os.path.join(texdir, "wrinkle_normal.png"), dilate(nrm * 0.5 + 0.5, m))
    _srgb_save(os.path.join(texdir, "wrinkle_mask.png"), dilate(np.clip(np.stack([mR, mG, mB, mA], -1), 0, 1), m))
    return {"normal": "tex/wrinkle_normal.png", "mask": "tex/wrinkle_mask.png", "drive": WRINKLE_DRIVE,
            "doc": " HEAD: mask RGBA = forehead / glabella / crow's feet / nasolabial; normal = the compressed state "
                   "(tangent space, OpenGL +Y, same UV as skin_head); blend = mask . clamp(sum w * shape weight)"}
