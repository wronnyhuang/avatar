"""asset — Juno skin textures in the MH hm08 UV layout (owner: skin-texture helper).

build(base, texdir) -> {"skin_head": material json, "skin_body": material json}  (+ PNGs in texdir, and _1k variants)

Everything is procedural and landmark/topology driven: the body-group UV triangles are rasterized (skin_raster) carrying
rest position, normal, AO, curvature and region masks, and every map is a function of those 3D fields, so the
textures follow the lead's design changes. Region masks come from MH targets (which vertices a target moves = a
topological region: ears, nostril wings, eyelids, cheeks), from landmarks (eyes, nose tip, mouth, joint cubes) and
from juno_design.scalp_mask. One texture set covers head + body (the hm08 atlas holds both); two materials point at it.

Maps (lod0 2048 — the head island alone is ~1300 px tall, i.e. ~0.2 mm/texel, finer than a 1080p close-up needs —
and lod1 1024 `_1k`):
  skin_albedo.png   sRGB  warm tan, soft colour zones, freckles, lips, nails, chestnut scalp under the hairline
  skin_normal.png   tangent space, OpenGL green-up: soft micro, lip lines, knuckle creases, nail edges
  skin_packed.png   linear RGBA: R AO, G roughness, B specular level (.5 = default F0), A SSS thickness (1 = thin)
"""
from __future__ import annotations

import os

import numpy as np
from PIL import Image
from scipy import ndimage
from scipy.spatial import cKDTree

from skin_raster import dilate, raster

N_TEX = 2048

# ---- palette (sRGB 0..255) -----------------------------------------------------------------------------------------
C_BASE = (200, 141, 120)       # warm light tan (lit ~ (205,140,105) per the lead)
C_WARM = (212, 118, 106)       # cheek / ear / nose flush
C_BLUSH = (214, 128, 112)      # asset-mouth4: face blush, peachier / less red than C_WARM
C_COOL = (192, 134, 116)       # jaw / upper lip
C_LID = (172, 112, 104)        # eyelid warm shadow
C_LIP = (182, 98, 96)        # lips: soft rose, low saturation
C_LIP_IN = (170, 78, 78)      # wet inner lip / mouth interior
C_NAIL = (238, 196, 184)      # nails: lighter pink
C_FRECKLE = (120, 70, 52)     # freckle pigment
C_ROOT = (62, 38, 27)         # dark chestnut hair root (scalp)


def s2l(c):
    c = np.asarray(c, np.float64) / 255.0
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def l2s(c):
    c = np.clip(c, 0, 1)
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * c ** (1 / 2.4) - 0.055)


def sstep(a, b, x):
    t = np.clip((x - a) / (b - a), 0, 1)
    return t * t * (3 - 2 * t)


def mix(a, b, t):
    t = np.asarray(t)[..., None] if np.ndim(t) else t
    return a * (1 - t) + b * t


# ---- 3D value noise (smooth, deterministic) ------------------------------------------------------------------------
class Noise3:
    def __init__(self, seed, size=64):
        self.g = np.random.default_rng(seed).random((size, size, size)).astype(np.float32)
        self.s = size

    def __call__(self, P, freq):
        x = P * freq
        i = np.floor(x).astype(np.int64)
        f = x - i
        f = f * f * (3 - 2 * f)
        s, g = self.s, self.g
        out = 0
        for dx in (0, 1):
            for dy in (0, 1):
                for dz in (0, 1):
                    w = (f[..., 0] if dx else 1 - f[..., 0]) * (f[..., 1] if dy else 1 - f[..., 1]) * \
                        (f[..., 2] if dz else 1 - f[..., 2])
                    out = out + w * g[(i[..., 0] + dx) % s, (i[..., 1] + dy) % s, (i[..., 2] + dz) % s]
        return out * 2 - 1   # -1..1

    def fbm(self, P, freq, oct=3):
        a, t, n = 1.0, 0.0, 0.0
        for k in range(oct):
            t = t + a * self(P + 17.3 * k, freq * 2 ** k)
            n += a
            a *= 0.5
        return t / n


# ---- mesh fields (per MH vertex) -----------------------------------------------------------------------------------
def _tris(base, faces):
    t = []
    for fi in faces:
        f = [v for v, _ in base.obj.faces[fi]]
        t.append(f[:3])
        if len(f) == 4:
            t.append([f[0], f[2], f[3]])
    return np.array(t, np.int64)


def _adjacency(tris, nV):
    e = np.concatenate([tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]])
    e = np.unique(np.sort(e, 1), axis=0)
    return e


def vertex_ao(base, body_tris):
    """Two-pass disc-based ambient occlusion (Bunnell 2005) on the skin, eyeballs as extra occluders."""
    V = base.VA
    occ_tris = np.concatenate([body_tris] + [_tris(base, base.faces_of(f"helper-{s}-eye")) for s in "lr"])
    fn = np.cross(V[occ_tris[:, 1]] - V[occ_tris[:, 0]], V[occ_tris[:, 2]] - V[occ_tris[:, 0]])
    fa = 0.5 * np.linalg.norm(fn, axis=1)
    area = np.zeros(base.nV)
    for k in range(3):
        np.add.at(area, occ_tris[:, k], fa / 3)
    nrm = np.zeros_like(V)
    for k in range(3):
        np.add.at(nrm, occ_tris[:, k], fn)
    nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-12)
    em = np.unique(occ_tris)
    rc = np.unique(body_tris)
    tree = cKDTree(V[em])
    K = 160
    d, j = tree.query(V[rc], k=K, distance_upper_bound=0.05)
    valid = np.isfinite(d) & (d > 1e-6)
    j = np.where(valid, j, 0)
    E = em[j]
    v = V[E] - V[rc][:, None]
    dd = np.maximum(np.linalg.norm(v, axis=2), 1e-6)
    vh = v / dd[..., None]
    cr = np.clip((nrm[rc][:, None] * vh).sum(2), 0, 1)
    ce = np.clip(-(nrm[E] * vh).sum(2), 0, 1) * 0.8 + 0.2   # back-facing occluders still block a little
    A = area[E]
    form = np.where(valid, A * cr * ce / (np.pi * dd ** 2 + A), 0)
    ao = np.ones(base.nV)
    ao1 = 1 - np.clip(form.sum(1), 0, 1)
    ao[rc] = ao1
    ao2 = 1 - np.clip((form * ao[E]).sum(1), 0, 1)
    ao[rc] = 0.5 * (ao1 + ao2)
    return ao


def vertex_curvature(base, edges, mask):
    """umbrella-Laplacian mean curvature (1/m): >0 concave (cavity), <0 convex."""
    V, N = base.VA, base.N
    s = np.zeros_like(V)
    c = np.zeros(base.nV)
    el = np.zeros(base.nV)
    for a, b in ((0, 1), (1, 0)):
        np.add.at(s, edges[:, a], V[edges[:, b]])
        np.add.at(c, edges[:, a], 1)
        np.add.at(el, edges[:, a], np.linalg.norm(V[edges[:, 0]] - V[edges[:, 1]], axis=1))
    c = np.maximum(c, 1)
    L = s / c[:, None] - V
    h = el / c
    k = (L * N).sum(1) / np.maximum(h * h, 1e-12) * 2
    k[~mask] = 0
    return k


def smooth_vertex(f, edges, it=2):
    for _ in range(it):
        s = f.copy()
        c = np.ones(len(f))
        np.add.at(s, edges[:, 0], f[edges[:, 1]])
        np.add.at(s, edges[:, 1], f[edges[:, 0]])
        np.add.at(c, edges[:, 0], 1)
        np.add.at(c, edges[:, 1], 1)
        f = s / c
    return f


def target_mask(base, rel, edges, lo=0.08, hi=0.5):
    m = np.linalg.norm(base.target(rel), axis=1)
    m = m / max(m.max(), 1e-12)
    return smooth_vertex(sstep(lo, hi, m), edges, 2)


# ---- lips: seam from the mouth-open target, vermilion as a clean analytic shape around it ---------------------------
def lip_frame(base, edges):
    V = base.VA
    mo = base.lm["mouth"]
    s = float(np.linalg.norm(base.lm["eye_c_l"] - base.lm["eye_c_r"]))
    op = np.linalg.norm(base.target("expression/units/caucasian/mouth-open.target.gz"), axis=1)
    near = (np.linalg.norm(V - mo, axis=1) < 0.7 * s) & base.skin_mask & (V[:, 2] > mo[2] - 0.006)
    om = op[near].max()
    up = np.where(near & (op < 0.35 * om))[0]
    lo = np.where(near & (op > 0.6 * om))[0]
    # upper and lower lip are separate surfaces that touch at the seam: pair each lower-lip vertex with the nearest
    # upper-lip vertex; close pairs straddle the seam
    dd, jj = cKDTree(V[up]).query(V[lo])
    ok = dd < 0.0018
    pts = 0.5 * (V[lo[ok]] + V[up[jj[ok]]])
    pts = pts[np.argsort(pts[:, 0])]
    w = float(np.abs(pts[:, 0]).max())
    # front-most seam y per x bin
    xb = np.linspace(-w, w, 25)
    sy = np.interp(xb, pts[:, 0], pts[:, 1])
    sy = 0.5 * (sy + sy[::-1])   # symmetric
    y0 = float(np.interp(0.0, xb, sy))
    mid = base.skin_mask & (np.abs(V[:, 0]) < 0.0035) & (np.abs(V[:, 1] - y0) < 0.5 * s) & (base.N[:, 2] > 0.3)
    mv = V[mid]
    upv = mv[mv[:, 1] > y0 + 0.0005]
    lov = mv[mv[:, 1] < y0 - 0.0005]
    yu = upv[np.argmax(upv[:, 2]), 1] if len(upv) else y0 + 0.006
    yl = lov[np.argmax(lov[:, 2]), 1] if len(lov) else y0 - 0.006
    hu = float(np.clip((yu - y0) * 1.35, 0.12 * s, 0.24 * s))
    hl = float(np.clip((y0 - yl) * 1.9, 0.16 * s, 0.32 * s))
    return dict(w=w, xb=xb, sy=sy, hu=hu, hl=hl, z=float(mo[2]), s=s)


def lip_mask(P, Nn, L):
    x, y = P[..., 0], P[..., 1]
    xn = np.clip(np.abs(x) / L["w"], 0, 1.5)
    dy = y - np.interp(x, L["xb"], L["sy"])
    base_u = (1 - np.clip(xn, 0, 1) ** 2) ** 0.65
    bow = 1 - 0.13 * np.exp(-(xn / 0.1) ** 2) + 0.07 * np.exp(-((xn - 0.24) / 0.12) ** 2)
    hu = L["hu"] * base_u * bow + 0.0004
    hl = L["hl"] * (1 - np.clip(xn, 0, 1) ** 2) ** 0.55 + 0.0004
    e = 0.00045   # soft border width (m)
    m_up = sstep(-e, e, hu - dy) * (dy > -0.002)
    m_lo = sstep(-e, e, hl + dy) * (dy <= 0.002)
    m = np.maximum(np.where(dy >= 0, m_up, 0), np.where(dy < 0, m_lo, 0))
    m *= sstep(1.04, 0.92, xn)
    m *= sstep(L["z"] - 0.016, L["z"] - 0.010, P[..., 2])  # not deep inside the mouth
    return m, dy, xn


# ---- hands: nails, knuckles, fingertips ----------------------------------------------------------------------------
def hand_fields(base, P):
    nail = np.zeros(P.shape[:-1], np.float32)
    nail_d = np.full(P.shape[:-1], 9.0, np.float32)   # signed distance to the nail border (m, <0 inside)
    knuck = np.zeros(P.shape[:-1], np.float32)
    crease = np.zeros(P.shape[:-1], np.float32)
    tip = np.zeros(P.shape[:-1], np.float32)
    finger = np.zeros(P.shape[:-1], np.float32)
    for side, sg in (("l", 1.0), ("r", -1.0)):
        J = {(i, k): base.cube(f"joint-{side}-finger-{i}-{k}") for i in range(1, 6) for k in range(1, 5)}
        lat = J[(2, 1)] - J[(5, 1)]
        lat /= np.linalg.norm(lat)
        dmid = J[(3, 4)] - J[(3, 1)]
        dmid /= np.linalg.norm(dmid)
        dors_hand = -sg * np.cross(dmid, lat)
        dors_hand /= np.linalg.norm(dors_hand)
        for i in range(1, 6):
            j3, j4 = J[(i, 3)], J[(i, 4)]
            d = j4 - j3
            Ld = np.linalg.norm(d)
            d /= Ld
            # thumb: its own dorsal = hand dorsal rotated about the finger axis toward the lateral side
            dors = dors_hand if i > 1 else (dors_hand * 0.35 + lat * 0.94)
            dors = dors - d * (dors @ d)
            dors /= np.linalg.norm(dors)
            side_ax = np.cross(d, dors)
            q = P - j3
            t = (q @ d) / Ld
            radial = q - np.outer(q.reshape(-1, 3) @ d, d).reshape(q.shape)
            r = np.linalg.norm(radial, axis=-1)
            rad = 0.0075 if i > 1 else 0.0085
            infinger = (r < 1.8 * rad) & (t > -0.3) & (t < 1.6)
            u = (radial @ dors)
            vlat = (radial @ side_ax)
            # nail plate: distal 55% of the last phalanx, dorsal, ~0.62 finger width
            ln, hw = 0.62, 0.55 * rad
            tc = 1.0 - ln / 2 + 0.08
            sd = np.maximum(np.abs(t - tc) * Ld - ln / 2 * Ld, np.abs(vlat) - hw)
            sd = np.where((u > 0.3 * rad) & infinger, sd, 9.0)
            nail_d = np.minimum(nail_d, sd)
            # fingertip warmth (distal pad + tip)
            tip = np.maximum(tip, infinger * sstep(0.3, 0.95, t))
            finger = np.maximum(finger, infinger * sstep(2.2 * rad, 1.2 * rad, r))
            # knuckles: dorsal over each joint 1..3, creases = lines across the finger at PIP/DIP
            for k in (1, 2, 3):
                jc = J[(i, k)]
                dk = (J[(i, k + 1)] - jc)
                dk /= np.linalg.norm(dk)
                qk = P - jc
                ax = qk @ dk
                rr = np.linalg.norm(qk - ax[..., None] * dk, axis=-1)
                dor = ((qk - ax[..., None] * dk) @ dors) / np.maximum(rr, 1e-9)
                kr = 0.011 if k == 1 else 0.008
                g = np.exp(-(ax / kr) ** 2) * sstep(-0.2, 0.6, dor) * (rr < 2.2 * rad)
                knuck = np.maximum(knuck, g * (1.0 if k > 1 else 0.7))
                if k > 1:
                    lines = sum(np.exp(-((ax - o) / 0.00045) ** 2) for o in (-0.0018, 0.0, 0.0017))
                    crease = np.maximum(crease, lines * sstep(0.1, 0.7, dor) * (rr < 1.6 * rad) *
                                        np.exp(-(ax / 0.004) ** 2))
    nail = sstep(0.0004, -0.0004, nail_d)
    return dict(nail=nail, nail_d=nail_d, knuck=knuck, crease=crease, tip=tip, finger=finger)


# ---- freckles ------------------------------------------------------------------------------------------------------
def freckles(base, P, region_v, rng, count=80, clusters=0, ymin=None):
    """soft freckles: positions sampled on the skin weighted by region_v (per vertex). clusters > 0 ( asset-mouth4,
    judge r4 "painted-on freckles, sparse uniform dots"): `clusters` cluster centres, each a small group of freckles with
    varied size / opacity (some overlapping -> merged blots), plus `count` scattered singles that are smaller and fainter."""
    V = base.VA
    w = np.clip(region_v, 0, None) ** 1.5
    cand = np.where(w > 0.05)[0]
    pick = rng.choice(cand, size=min(count, len(cand)), replace=False, p=w[cand] / w[cand].sum())
    # jitter along the surface (tangent offset ~1.5 mm, re-snap later via 3D distance)
    pts = V[pick] + rng.normal(0, 0.0012, (len(pick), 3))
    rad = rng.uniform(0.0007, 0.0014, len(pick)) * np.where(rng.random(len(pick)) < 0.15, 1.5, 1.0)   # ASSET2: read at card size
    amt = rng.uniform(0.6, 1.0, len(pick))
    if clusters:
        rad *= 0.8
        amt *= rng.uniform(0.5, 0.9, len(pick))
        wc = w ** 1.5 * (1.0 if ymin is None else (V[:, 1] > ymin))   # clusters: the dense core only (no blots by the mouth)
        cand2 = np.where(wc > 0.02)[0]
        cc = rng.choice(cand2, size=min(clusters, len(cand2)), replace=False, p=wc[cand2] / wc[cand2].sum())
        cp, cr, ca = [], [], []
        for c in cc:
            k = rng.integers(3, 11)
            spread = rng.uniform(0.0015, 0.0032)
            cp.append(V[c] + rng.normal(0, spread, (k, 3)))
            r = rng.lognormal(np.log(0.0010), 0.35, k)
            r[rng.random(k) < 0.12] *= 1.6                       # an occasional larger blot
            cr.append(r)
            ca.append(rng.uniform(0.55, 1.0, k) * rng.uniform(0.75, 1.0))   # per-cluster density
        pts = np.vstack([pts] + cp)
        rad = np.concatenate([rad] + cr)
        amt = np.concatenate([amt] + ca)
    F = np.zeros(P.shape[:-1], np.float32)
    flat = P.reshape(-1, 3)
    idx = np.where(np.isfinite(flat[:, 0]))[0]
    tree = cKDTree(flat[idx])
    out = F.reshape(-1)
    for p, r, a in zip(pts, rad, amt):
        ii = tree.query_ball_point(p, 3 * r)
        if not ii:
            continue
        ii = idx[np.array(ii)]
        d = np.linalg.norm(flat[ii] - p, axis=1)
        f = a * np.exp(-(d / r) ** 2)
        out[ii] = out[ii] + f - out[ii] * f if clusters else np.maximum(out[ii], f)   # soft union: overlaps merge
    return F


REF_BLEND = False   # OFF: v4 test came out far too dark (value-range/colour-space mismatch to resolve first)


def _blend_reference(albedo, n, M):
    """SCULPT's reference albedo baked on our head UVs (apply_design_fit.ref_textures): mask*ref*gain + (1-mask)*ours.
    gain matches the reference's mean to OUR palette inside the mask (the target tone is the concept's warm light tan)."""
    try:
        import apply_design_fit
        rt = apply_design_fit.ref_textures()
    except Exception:
        return albedo
    from PIL import Image as _I
    ref = np.asarray(_I.open(rt["albedo"]).convert("RGB").resize((n, n), _I.BILINEAR), np.float64) / 255.0
    m = np.asarray(_I.open(rt["mask"]).convert("L").resize((n, n), _I.BILINEAR), np.float64) / 255.0
    m = m * M
    sel = m > 0.5
    if sel.sum() < 100:
        return albedo
    a = albedo[..., :3]
    gain = np.clip(a[sel].mean(0) / np.maximum(ref[sel].mean(0), 1e-3), 0.8, 1.8)
    out = albedo.copy()
    out[..., :3] = m[..., None] * np.clip(ref * gain, 0, 1) + (1 - m[..., None]) * a
    return out


# ---- main ----------------------------------------------------------------------------------------------------------
def build(base, texdir):
    import juno_design
    rng = np.random.default_rng(4430)
    n = N_TEX
    faces = base.faces_of("body")
    body_tris = _tris(base, faces)
    edges = _adjacency(body_tris, base.nV)
    R = raster(base, faces, n)
    M = R.mask
    V = base.VA

    # vertex fields
    ao_v = vertex_ao(base, body_tris)
    curv_v = smooth_vertex(vertex_curvature(base, edges, base.skin_mask), edges, 1)
    scalp_v = juno_design.scalp_mask(base)
    ear_v = np.maximum(target_mask(base, "ears/l-ear-scale-incr.target.gz", edges, 0.15, 0.6),
                       target_mask(base, "ears/r-ear-scale-incr.target.gz", edges, 0.15, 0.6))
    ala_v = target_mask(base, "nose/nose-flaring-incr.target.gz", edges, 0.1, 0.6)
    lid_v = np.maximum(target_mask(base, "expression/units/caucasian/eye-left-closure.target.gz", edges, 0.05, 0.4),
                       target_mask(base, "expression/units/caucasian/eye-right-closure.target.gz", edges, 0.05, 0.4))
    cheek_v = np.maximum(target_mask(base, "cheek/l-cheek-volume-incr.target.gz", edges, 0.05, 0.7),
                         target_mask(base, "cheek/r-cheek-volume-incr.target.gz", edges, 0.05, 0.7))

    # texel fields
    P = R.interp(V)
    Nn = R.interp(base.N)
    Nn /= np.maximum(np.linalg.norm(Nn, axis=-1, keepdims=True), 1e-9)
    ao = R.interp(ao_v)[..., 0]
    curv = R.interp(curv_v)[..., 0]
    scalp = R.interp(scalp_v)[..., 0]
    ear = R.interp(ear_v)[..., 0]
    ala = R.interp(ala_v)[..., 0]
    lid = R.interp(lid_v)[..., 0]
    cheek = R.interp(cheek_v)[..., 0]
    Pn = np.where(M[..., None], P, np.nan)

    lm = base.lm
    s = float(np.linalg.norm(lm["eye_c_l"] - lm["eye_c_r"]))           # face scale (inter-pupil)
    ecl, ecr, nose, mouth, chin = lm["eye_c_l"], lm["eye_c_r"], lm["nose_tip"], lm["mouth"], lm["chin"]
    head_y = lm["neck"][1]

    def g3(c, r):
        return np.exp(-((P - c) ** 2).sum(-1) / r ** 2)

    front = sstep(-0.1, 0.4, Nn[..., 2])
    headm = sstep(head_y - 0.01, head_y + 0.03, P[..., 1])

    # islands: mouth interior & the eyelid tuck (hm08 has separate UV islands for them)
    lab_tri = np.zeros(len(R.tv), np.int32)
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    rr = np.repeat(np.arange(len(R.tt)), 3)
    A = coo_matrix((np.ones(rr.size), (rr, R.tt.reshape(-1)))).tocsr()
    _, lab_tri = connected_components(A @ A.T)
    tri_c = V[R.tv].mean(1)
    lab_tex = np.where(M, lab_tri[np.maximum(R.tri, 0)], -1)
    isl_mouth = np.zeros_like(M)
    isl_eye = np.zeros_like(M)
    for k in np.unique(lab_tri):
        sel = lab_tri == k
        c = tri_c[sel].mean(0)
        if sel.sum() > 0.2 * len(lab_tri):
            continue
        if np.linalg.norm(c - mouth) < 0.5 * s:
            isl_mouth |= lab_tex == k
        elif min(np.linalg.norm(c - ecl), np.linalg.norm(c - ecr)) < 0.5 * s:
            isl_eye |= lab_tex == k

    # lips
    L = lip_frame(base, edges)
    _lip_geo, ldy, lxn = lip_mask(P, Nn, L)
    # lip colour region = the hm08 vermilion by TOPOLOGY (MPFB2 CC0 mpfb_lips.jpg UV mask, same atlas layout), softened;
    # the analytic lip_mask over-painted thin fitted lips into a pout (lead review v1/v2).
    from PIL import Image as _I, ImageFilter as _F
    _lm = _I.open(os.path.join(base.data, "textures", "mpfb_lips.jpg")).convert("L").resize((n, n), _I.BILINEAR)
    _lm = _lm.filter(_F.GaussianBlur(radius=max(1.0, 1.5 * n / 2048)))
    lip = np.clip((np.asarray(_lm, np.float32) / 255.0 - 0.25) / 0.5, 0, 1)
    lip = np.where(isl_mouth, 1.0, lip) * headm

    # hands
    H = hand_fields(base, P)

    # ------------------------------------------------------------------ albedo (linear)
    nz = Noise3(11)
    base_c = s2l(C_BASE)
    col = np.broadcast_to(base_c, P.shape).astype(np.float64).copy()
    low = nz.fbm(np.nan_to_num(P), 18.0, 3)                 # ~5 cm blotches, very soft
    col *= (1 + 0.035 * low)[..., None]
    # cool jaw / upper lip / chin
    philtrum = g3(mouth + np.array([0, L["hu"] + 0.004, 0.0]), 0.35 * s)
    jaw = np.clip(g3(chin, 0.75 * s) + 0.6 * philtrum, 0, 1) * headm
    col = mix(col, s2l(C_COOL), 0.35 * jaw)
    # warm zones: cheeks (target region, centred under the eyes), nose tip, ears
    # asset-mouth4 (judge r4 "blotchy red/pink flush on cheeks and nose"): a wider, softer, less red blush (a
    # peach blood-flow tint that fades smoothly into the base, no hard-edged target region), nose much lighter
    ck_l = g3(ecl + np.array([0.15 * s, -0.78 * s, 0.0]), 0.68 * s)
    ck_r = g3(ecr + np.array([-0.15 * s, -0.78 * s, 0.0]), 0.68 * s)
    flush_cheek = np.clip(0.25 * cheek + ck_l + ck_r, 0, 1) ** 1.3 * front * headm
    flush_nose = np.clip(g3(nose, 0.42 * s) + 0.4 * ala, 0, 1) * headm
    flush = np.clip(0.52 * flush_cheek + 0.34 * flush_nose + 0.32 * ear, 0, 1)   # v6: ears less red; mouth4 .68/.6 -> .42/.30
    col = mix(col, s2l(C_BLUSH), flush)
    # eyelids: warm shadow
    lidz = np.clip(lid + 0.6 * np.exp(-(np.minimum(np.linalg.norm(P - ecl, axis=-1),
                                                     np.linalg.norm(P - ecr, axis=-1)) / (0.62 * s)) ** 2 * 1.0) *
                   (P[..., 1] > ecl[1] - 0.12 * s), 0, 1) * headm
    # asset-eyes3: the warm lid shadow is an UPPER-lid tint only (it was smudging the lower lid + under-eye into a
    # dark crescent): fade it out from just above the eye centre down
    ey = 0.5 * (ecl[1] + ecr[1])
    upper_g = sstep(ey - 0.02 * s, ey + 0.10 * s, P[..., 1])
    lidz = lidz * upper_g
    col = mix(col, s2l(C_LID), 0.4 * lidz)
    # under-eye clean zone (lower lid skin + the cheek top under each eye): baked AO in the albedo/packed is lifted here
    # (the eyeball occluders darkened it into "eye bags")
    und = np.clip(np.exp(-((P[..., 0] - ecl[0]) ** 2 + ((P[..., 1] - (ecl[1] - 0.22 * s)) / 0.8) ** 2) / (0.36 * s) ** 2) +
                  np.exp(-((P[..., 0] - ecr[0]) ** 2 + ((P[..., 1] - (ecr[1] - 0.22 * s)) / 0.8) ** 2) / (0.36 * s) ** 2),
                  0, 1) * (1 - upper_g) * headm * front
    # asset-eyes3: inner-lid tuck closer to the lid skin (the saturated pink read as a red outline ring at grazing
    # angles: far eye in 3/4, ragged reddish fringe between the lashes)
    # asset-eyes4: the tuck IS the lid rim/waterline: upper rim = lid-skin tone, a touch darker/pinker (the lid's
    # thickness reads as a shelf, not an outline); lower = a moist pink waterline (low roughness below -> a spec line)
    rim_lo = sstep(ey + 0.04 * s, ey - 0.04 * s, P[..., 1])
    col = np.where(isl_eye[..., None], mix(s2l((204, 138, 124)), s2l((224, 150, 140)), rim_lo), col)
    # body warm joints (elbows, knees) and hands
    for jn in ("joint-l-elbow", "joint-r-elbow", "joint-l-knee", "joint-r-knee"):
        col = mix(col, s2l(C_WARM), 0.25 * g3(base.cube(jn), 0.045))
    col = mix(col, s2l(C_WARM), np.clip(0.35 * H["knuck"] + 0.4 * H["tip"], 0, 0.55))
    # freckles: nose bridge + upper cheeks
    bridge = nose * 0.45 + 0.5 * (ecl + ecr) * 0.55
    fr_region = (np.exp(-((V - bridge) ** 2).sum(1) / (0.4 * s) ** 2) +
                 0.9 * np.exp(-((V - (ecl + np.array([0.1 * s, -0.6 * s, 0.05 * s]))) ** 2).sum(1) / (0.42 * s) ** 2) +
                 0.9 * np.exp(-((V - (ecr + np.array([-0.1 * s, -0.6 * s, 0.05 * s]))) ** 2).sum(1) / (0.42 * s) ** 2))
    fr_region *= (base.N[:, 2] > 0.35) & base.skin_mask
    fr_region *= 1 - np.clip(lid_v * 2, 0, 1)
    fr = freckles(base, Pn, fr_region, rng, count=70, clusters=50, ymin=float(nose[1]) - 0.02 * s)   # mouth4: clustered (was 150 singles)
    col = mix(col, col * s2l(C_FRECKLE) / s2l(C_BASE), 0.5 * fr)   # mouth4: .85 -> .72; ASSET3 v7: .5 (b4 read as dirt smudges at card size)
    # lips
    lip_c = mix(np.broadcast_to(s2l(C_LIP), col.shape), s2l(C_LIP_IN), sstep(L["z"] - 0.002, L["z"] - 0.010, P[..., 2]))
    col = mix(col, lip_c, lip)
    # ASSET2: lighter lower-lip centre (soft highlight pigment) + a slightly deeper lip border
    llc = g3(mouth + np.array([0.0, -0.55 * L['hl'], 0.002]), 0.22 * s) * lip
    col = mix(col, s2l((204, 128, 122)), 0.25 * np.clip(llc, 0, 1))
    edge = np.clip(lip * (1 - lip) * 4, 0, 1)
    col = mix(col, s2l((150, 82, 80)), 0.25 * edge)
    # asset-mouth3 (judge r2 "lips flat/blurry, no lip detail"): crisp vermilion border = a darker rim just inside
    # the vermilion + a pale "white roll" just outside it; a little more saturation in the lip body
    lip_in = sstep(0.45, 0.75, lip) * (1 - sstep(0.75, 0.95, lip)) * (1 - isl_mouth)
    col = mix(col, s2l((160, 84, 84)), 0.30 * lip_in)
    wroll = np.exp(-((lip - 0.18) / 0.12) ** 2) * (lip < 0.45) * (1 - isl_mouth) * headm
    col = mix(col, col * 1.10, np.clip(wroll, 0, 1))
    col = mix(col, s2l((176, 88, 90)), 0.20 * sstep(0.85, 1.0, lip) * (1 - isl_mouth))
    # asset-mouth4: subtle lip colour variation -- faint vertical lip lines in the pigment (same phase as the normal
    # map's lines), a lighter lower-lip centre, slightly deeper corners
    _lmx, _lmy = lxn, ldy
    _lines = np.sin(np.nan_to_num(P[..., 0]) / 0.0011 * 2 * np.pi + 1.7 * nz(np.nan_to_num(P), 300.0)) * 0.5 + 0.5
    lipv = lip * (1 - isl_mouth)
    col *= (1 - 0.06 * (_lines ** 3) * lipv * sstep(0.95, 0.4, _lmx))[..., None]
    llc2 = np.exp(-(_lmx / 0.38) ** 2) * np.exp(-((_lmy + 0.5 * L['hl']) / (0.45 * L['hl'])) ** 2) * lipv
    col = mix(col, s2l((214, 140, 132)), 0.22 * np.clip(llc2, 0, 1))
    col = mix(col, s2l((146, 74, 74)), 0.30 * sstep(0.68, 0.98, _lmx) * lipv)
    # nails
    col = mix(col, s2l(C_NAIL), H["nail"])
    # scalp: hair-root chestnut under the hairline, soft gradient
    sc = sstep(0.1, 0.65, np.clip(scalp + 0.12 * nz(np.nan_to_num(P), 900.0), 0, 1))
    col = mix(col, s2l(C_ROOT), sc)
    # asset-mouth4 (judge r4 "crude pasted-on ears"): ear folds read -- concave folds (concha, scapha, the groove
    # inside the helix) darker and warmer, convex ridges (helix rim, antihelix) a touch lighter
    ear_m = np.clip(ear * 1.6, 0, 1)
    fold = sstep(10.0, 120.0, curv) * ear_m
    ridge = sstep(-10.0, -120.0, curv) * ear_m
    col = mix(col, col * np.array([0.62, 0.52, 0.50]), 0.55 * fold)
    col *= (1 + 0.08 * ridge)[..., None]
    # gentle AO in the albedo (crevices read in a soft animated-film way, not dirty)
    col *= (0.82 + 0.18 * mix(ao[..., None], np.ones_like(ao[..., None]), 0.85 * und)[..., 0])[..., None]
    albedo = l2s(col)
    if REF_BLEND:
        albedo = _blend_reference(albedo, n, M)

    # ------------------------------------------------------------------ height (m) -> tangent-space normal
    Pz = np.nan_to_num(P)
    micro = Noise3(23)
    h = 0.000006 * micro.fbm(Pz, 2500.0, 2) * (1 - sc)                    # very soft fine undulation (~0.4 mm)
    h += 0.000004 * micro.fbm(Pz + 3.1, 700.0, 2)
    # lip vertical lines (follow x), stronger in the middle of each lip, fading at the border
    # asset-mouth3: phase from SIGNED x (was |x|: the cusp of |x| at the midline made a mirrored line pair = a dark
    # vertical crease in the lower-lip centre in the normal map)
    lines = np.sin(Pz[..., 0] / 0.0011 * 2 * np.pi + 1.7 * nz(Pz, 300.0)) * 0.5 + 0.5
    h -= 0.000035 * (lines ** 3) * lip * (1 - isl_mouth) * sstep(0.95, 0.4, lxn)
    # asset-mouth3: vermilion volume (a soft step up onto the lip -> a crisp border line in the shading), the white
    # roll ridge just outside the border, philtrum columns + groove from the cupid's bow peaks to the nose base
    lipc = lip * (1 - isl_mouth)
    h += 0.00022 * sstep(0.25, 0.8, lipc)
    h += 0.00010 * np.exp(-((lip - 0.15) / 0.10) ** 2) * (lip < 0.4) * (1 - isl_mouth)
    px, py = Pz[..., 0], Pz[..., 1]
    yb = float(np.interp(0.0, L["xb"], L["sy"])) + 0.87 * L["hu"]       # upper border at the centre (bow dip)
    yn = yb + 0.20 * s                                                   # ~ nose base
    t_ph = np.clip((py - yb) / (yn - yb), 0, 1)
    ph_y = sstep(-0.05, 0.12, (py - yb) / (yn - yb)) * sstep(1.05, 0.7, (py - yb) / (yn - yb)) * front * headm
    xc = 0.24 * L["w"] * (1 - 0.35 * t_ph)                              # columns converge a little toward the nose
    col_r = np.exp(-((np.abs(px) - xc) / (0.09 * L["w"])) ** 2)
    groove = np.exp(-(px / (0.10 * L["w"])) ** 2)
    h += (0.00030 * col_r - 0.00014 * groove) * ph_y * (1 - lipc)
    # knuckle creases
    h -= 0.00007 * np.clip(H["crease"], 0, 1)   # ASSET2: deeper knuckle creases
    # nail: raised plate with a crisp edge
    h += 0.00010 * sstep(0.0005, -0.0003, H["nail_d"]) * (H["nail_d"] < 1)
    h = np.where(M, h, 0)
    h = dilate(h, M)
    Pd = dilate(np.nan_to_num(P), M)
    dPdx = np.gradient(Pd, axis=1)   # per texel along +u
    dPdy = -np.gradient(Pd, axis=0)  # per texel along +v (image up)
    lu = np.maximum(np.linalg.norm(dPdx, axis=-1), 1e-7)
    lv = np.maximum(np.linalg.norm(dPdy, axis=-1), 1e-7)
    dhu = np.gradient(h, axis=1) / lu
    dhv = -np.gradient(h, axis=0) / lv
    # sign of the bitangent relative to dP/dv (mirrored islands): tangent = dP/du, bitangent = cross(N,T)*w
    Tn = dPdx / lu[..., None]
    Bn = np.cross(Nn, Tn)
    wsign = np.where((Bn * dPdy).sum(-1) < 0, -1.0, 1.0)
    nts = np.stack([-dhu, -dhv * wsign, np.ones_like(h)], -1)
    nts /= np.linalg.norm(nts, axis=-1, keepdims=True)
    nts = np.where(M[..., None], nts, np.array([0, 0, 1.0]))
    normal = nts * 0.5 + 0.5

    # ------------------------------------------------------------------ packed
    cav = sstep(0, 60.0, curv)                                    # concave creases
    ao_t = np.clip(ao * (1 - 0.25 * cav), 0, 1)
    ao_t = np.where(isl_mouth, ao_t * 0.55, ao_t)
    ao_t = ao_t + (1 - ao_t) * 0.7 * und * (1 - isl_eye)        # asset-eyes3: clean under-eye (no baked bag)
    tzone = np.clip(g3(nose, 0.55 * s) + g3(0.5 * (ecl + ecr) + np.array([0, 0.9 * s, 0.25 * s]), 0.6 * s) +
                    g3(chin, 0.4 * s), 0, 1) * headm
    rough = np.where(headm > 0.5, 0.5, 0.55) - 0.12 * tzone
    rough = mix(rough[..., None], np.array([0.33]), lip)[..., 0]
    rough = np.where(isl_mouth | isl_eye, 0.25, rough)
    rough = np.where(isl_eye, 0.32 - 0.2 * rim_lo, rough)          # asset-eyes4: wet lower waterline, satin upper rim
    rough = mix(rough[..., None], np.array([0.3]), H["nail"])[..., 0]
    rough = mix(rough[..., None], np.array([0.75]), sc)[..., 0]    # scalp: never shiny between strands
    rough += 0.03 * micro(Pz, 400.0)
    spec = 0.5 + 0.06 * lip - 0.35 * sc + 0.1 * H["nail"]
    thick = 0.12 + 0.5 * np.clip(ear, 0, 1)   # v6: less red ear transmission
    thick = np.maximum(thick, 0.7 * ala)
    thick = np.maximum(thick, 0.65 * lidz)
    thick = np.maximum(thick, 0.65 * H["finger"])
    thick = np.maximum(thick, 0.45 * lip)
    thick = np.maximum(thick, 0.3 * g3(nose, 0.25 * s))
    packed = np.stack([ao_t, np.clip(rough, 0.05, 1), np.clip(spec, 0, 1), np.clip(thick, 0, 1)], -1)

    # ------------------------------------------------------------------ write (dilate across seams)
    out = {}
    for nm, img in (("skin_albedo", albedo), ("skin_normal", normal), ("skin_packed", packed)):
        img = dilate(np.where(M[..., None], img, 0), M)
        a8 = np.clip(np.round(img * 255), 0, 255).astype(np.uint8)
        mode = "RGBA" if a8.shape[-1] == 4 else "RGB"
        im = Image.fromarray(a8, mode)
        im.save(os.path.join(texdir, nm + ".png"), optimize=False, compress_level=6)
        im1 = im.resize((n // 2, n // 2), Image.LANCZOS)
        if nm == "skin_normal":   # renormalize after filtering
            a = np.asarray(im1, np.float64) / 255 * 2 - 1
            a /= np.maximum(np.linalg.norm(a, axis=-1, keepdims=True), 1e-9)
            im1 = Image.fromarray(np.clip(np.round((a * 0.5 + 0.5) * 255), 0, 255).astype(np.uint8))
        im1.save(os.path.join(texdir, nm + "_1k.png"), compress_level=6)
        out[nm] = f"tex/{nm}.png"
    common = {"type": "skin", "albedo": out["skin_albedo"], "normal": out["skin_normal"], "packed": out["skin_packed"]}
    mats = {
        "skin_head": dict(common, params={"sss_color": [0.88, 0.40, 0.30], "sss_radius_mm": [3.8, 1.5, 0.75],
                                          "spec_f0": 0.028, "normal_strength": 1.0}),
        "skin_body": dict(common, params={"sss_color": [0.86, 0.42, 0.31], "sss_radius_mm": [3.4, 1.35, 0.7],
                                          "spec_f0": 0.027, "normal_strength": 1.0}),
    }
    return mats
