"""HEAD-MTEX — procedural mouth maps for junor: teeth / gums / tongue / mouth_interior (albedo + normal + packed).

Every map is painted per texel from the rest geometry (no images, no licences):
  paint(name, parts, fn, texdir, n, ...)   parts = [(part_name, P (nV,3), UV (nV,2), tris (nT,3), N (nV,3)), ...]
  fn(C) -> dict(col (n,n,3) sRGB 0..1, ao, rough, spec, thick, height (metres, optional))
  C = per-texel context: P, N (interpolated rest position / normal), tri, comp (connected component id = one tooth),
      part (index into parts), mask, tm (metres per texel), rng, extra per-material fields.
packed = R AO, G roughness, B spec level (.5 = default), A SSS thickness (juno.asset/1).

What SHADE's mat_mouth_pbr.glsl does with them (so the maps are made FOR it, not against it):
  teeth   alb' = (0.7 alb + 0.3 mean(alb)) * (0.92, 0.86, 0.76), clamped at 0.62 linear  -> the albedo is painted
          near-NEUTRAL (the shader warms it) and below ~0.66 linear (sRGB ~212) so it is not clamped to flat grey
          (v1-v6: enamel 238/230/212 clamped every channel -> a uniform 0.62 grey); packed AO is S.ao (room light +
          spec occlusion); roughness is the uniform uPbrMouth.w (packed G ignored)
  gums / tongue / mouth_interior  alb = max(alb, tissue_min) toward the centre only (the buccal sides keep the
          asset's dark albedo); normal maps are used (S.n); roughness constants in the shader.
"""
from __future__ import annotations

import os

import numpy as np
from PIL import Image
from scipy import ndimage
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from bake import dilate, raster
import tex_r


def _sm(a, b, x):
    t = np.clip((x - a) / (b - a), 0, 1)
    return t * t * (3 - 2 * t)


def _comps(nv, tris):
    r = np.concatenate([tris[:, 0], tris[:, 1], tris[:, 2]])
    c = np.concatenate([tris[:, 1], tris[:, 2], tris[:, 0]])
    return connected_components(coo_matrix((np.ones(len(r)), (r, c)), shape=(nv, nv)))[1]


def _noise(rng, sh, sig):
    a = ndimage.gaussian_filter(rng.standard_normal(sh), sig)
    return a / max(a.std(), 1e-9)


def _surf(P, T, step=0.00025):
    """dense points on a triangle surface (~step apart): nearest-VERTEX distances give dotted shadow lines."""
    out = [P[np.unique(T)]]
    a, b, c = P[T[:, 0]], P[T[:, 1]], P[T[:, 2]]
    L = np.maximum(np.maximum(np.linalg.norm(b - a, axis=1), np.linalg.norm(c - a, axis=1)), np.linalg.norm(c - b, axis=1))
    k = np.clip(np.ceil(L / step).astype(int), 1, 24)
    for kk in np.unique(k):
        sel = k == kk
        i, j = np.meshgrid(np.arange(kk + 1), np.arange(kk + 1))
        ok = (i + j) <= kk
        u, v = i[ok] / kk, j[ok] / kk
        A, B, Cc = a[sel], b[sel], c[sel]
        pts = A[:, None] + (B - A)[:, None] * u[None, :, None] + (Cc - A)[:, None] * v[None, :, None]
        out.append(pts.reshape(-1, 3))
    return np.concatenate(out)


class Ctx:
    pass


def paint(name, parts, fn, texdir, n=1024, spec=0.5, thick=0.5, typ=None, params=None, seed=1, save=None):
    uvt, Pc, Nc, part_c, comp_c = [], [], [], [], []
    cbase, cb = 0, []
    for k, (pn, P, UV, T, N) in enumerate(parts):
        lab = _comps(len(P), T)
        uvt.append(UV[T]); Pc.append(P[T]); Nc.append(N[T])
        part_c.append(np.full(len(T), k)); comp_c.append(lab[T[:, 0]] + cbase)
        cb.append(cbase)
        cbase += lab.max() + 1
    uvt, Pc, Nc = np.concatenate(uvt), np.concatenate(Pc), np.concatenate(Nc)
    part_c, comp_c = np.concatenate(part_c), np.concatenate(comp_c)
    R = raster(uvt, n, n)
    C = Ctx()
    C.mask = m = R.mask
    C.P = R.interp(Pc)
    Nn = R.interp(Nc)
    C.N = Nn / np.maximum(np.linalg.norm(Nn, axis=-1, keepdims=True), 1e-9)
    C.tri = R.tri
    C.part = np.where(m, part_c[np.maximum(R.tri, 0)], -1)
    C.comp = np.where(m, comp_c[np.maximum(R.tri, 0)], -1)
    C.tm = tex_r._texel_m(uvt, Pc, n)
    C.rng = np.random.default_rng(seed)
    C.n = n
    C.parts = parts
    C.cbase = cb
    r = fn(C)
    sh = (n, n)
    col = r["col"]
    ao = r.get("ao", np.ones(sh))
    rough = r.get("rough", np.full(sh, 0.4))
    sp = r.get("spec", np.full(sh, spec))
    th = r.get("thick", np.full(sh, thick))
    h = r.get("height")
    alb = dilate(np.concatenate([np.clip(col, 0, 1), np.ones((n, n, 1))], -1), m)
    pk = dilate(np.stack([np.clip(ao, 0, 1), np.clip(rough, 0.05, 1), np.clip(sp, 0, 1), np.clip(th, 0, 1)], -1), m)
    if h is not None:
        h = dilate(h * m, m)
        nrm = dilate(tex_r._normal_from_height(h, C.tm) * 0.5 + 0.5, m)
    else:
        nrm = np.zeros((64, 64, 3))
        nrm[...] = (0.5, 0.5, 1.0)
    save = save or tex_r._srgb_save
    for key, img in (("albedo", alb), ("packed", pk), ("normal", nrm)):
        save(os.path.join(texdir, f"{name}_{key}.png"), img)
    return {"type": typ or name, "albedo": f"tex/{name}_albedo.png", "normal": f"tex/{name}_normal.png",
            "packed": f"tex/{name}_packed.png", "params": params or {}}


# ------------------------------------------------------------------ teeth
def teeth_fn(mouth_lm):
    m = np.asarray(mouth_lm, float)

    def fn(C):
        sh = (C.n, C.n)
        P, msk = C.P, C.mask
        x, y, z = P[..., 0], P[..., 1], P[..., 2]
        e = np.zeros(sh)          # distance from the biting edge (m)
        hf = np.zeros(sh)         # 0 edge .. 1 top of the mesh
        gap = np.full(sh, 1.0)    # distance to the neighbouring tooth (m)
        md = np.zeros(sh)         # mesio-distal position across the tooth (-1..1)
        tone = np.zeros((C.n, C.n, 3))
        rng = C.rng
        # per arch: the gum surface of that arch (cervical line = where the tooth meets the gum)
        gum_tree = getattr(C, "gum_trees", {})
        dg = np.full(sh, 1.0)
        for k, (pn, Pv, UV, T, N) in enumerate(C.parts):
            upper = "upper" in pn
            lab = _comps(len(Pv), T)
            sel_k = C.part == k
            comp_k = C.comp - C.cbase[k]
            trees = {}
            for t in range(lab.max() + 1):
                tt_ = T[lab[T[:, 0]] == t]
                trees[t] = cKDTree(_surf(Pv, tt_))
            for t in range(lab.max() + 1):
                s = sel_k & (comp_k == t)
                if not s.any():
                    continue
                pv = Pv[lab == t]
                yv = pv[:, 1]
                ed = (y[s] - yv.min()) if upper else (yv.max() - y[s])
                e[s] = ed
                hf[s] = ed / max(np.ptp(yv), 1e-4)
                q = P[s]
                d = np.full(len(q), 1.0)
                for t2, tr in trees.items():
                    if t2 != t:
                        d = np.minimum(d, tr.query(q, distance_upper_bound=0.003)[0])
                gap[s] = d
                # mesio-distal coordinate across the tooth face (-1..1): the arch tangent ~ perpendicular to the
                # centroid's direction from the arch centre (0, ., z_back)
                cxz = np.array([pv[:, 0].mean(), pv[:, 2].mean()])
                rad = cxz - np.array([0.0, Pv[:, 2].min()])
                tan = np.array([rad[1], -rad[0]]) / max(np.linalg.norm(rad), 1e-9)
                proj = (pv[:, [0, 2]] - cxz) @ tan
                hw = max(np.percentile(np.abs(proj), 95), 1e-4)
                md[s] = ((q[:, [0, 2]] - cxz) @ tan) / hw
                # per-tooth tone: a few % value + a little more yellow on the canines / premolars
                cx = abs(pv[:, 0].mean())
                yel = 0.5 * _sm(0.008, 0.013, cx) * _sm(0.026, 0.018, cx)
                tone[s] = (1 + rng.normal(0, 0.025)) * np.array([1.0, 1.0 - 0.02 * yel, 1.0 - 0.07 * yel])
            gt = gum_tree.get("upper" if upper else "lower")
            if gt is not None:
                dg[sel_k] = np.minimum(gt.query(P[sel_k], distance_upper_bound=0.006)[0], 1.0)
        front = _sm(0.016, 0.008, np.abs(x))                  # incisors / canines
        # palette (sRGB): near-neutral because SHADE warms the teeth by (0.92, 0.86, 0.76)
        body = np.array([211, 210, 204]) / 255
        incis = np.array([136, 150, 172]) / 255                 # thin enamel edge: greyer, a little blue (translucent)
        cerv = np.array([206, 186, 146]) / 255                  # dentine shows through near the neck: warm yellow
        dark = np.array([118, 104, 92]) / 255                   # between the teeth
        k_inc = (_sm(0.0024, 0.0003, e) * (0.35 + 0.65 * front))[..., None]
        # mamelon hint: the translucent edge band is wavy across each incisor
        k_inc = k_inc * (0.85 + 0.15 * _noise(rng, sh, 2.0)[..., None].clip(-1, 1))
        k_cerv = np.clip(_sm(0.003, 0.0006, dg) * 0.7, 0, 1)[..., None]
        col = body * (1 - k_inc) + incis * k_inc
        col = col * (1 - k_cerv) + cerv * k_cerv
        col = col * tone
        # faint enamel mottling + vertical developmental lobes on the incisors
        mot = _noise(rng, sh, max(0.6e-3 / C.tm, 1.5))
        col = col * (1 + 0.012 * mot[..., None])
        # developmental lobes: three faint vertical lobes on the incisor faces (lighter ridges, two soft grooves)
        lobe = (np.cos(np.clip(md, -1, 1) * 1.5 * np.pi) * _sm(0.010, 0.002, e) * front)[..., None]
        col = col * (1 + 0.035 * lobe)
        # interproximal: the contact gaps go dark (light does not get between the teeth)
        k_gap = _sm(0.0009, 0.00005, gap)[..., None] * (1 - 0.3 * k_inc)
        col = col * (1 - 0.55 * k_gap) + dark * 0.55 * k_gap
        # gum-line shadow (the gum margin overhangs the neck a little)
        k_gl = _sm(0.0012, 0.0, dg)[..., None]
        col = col * (1 - 0.35 * k_gl)
        # back teeth: less light, slightly darker albedo too (as v5, keeps the cavity reading)
        back = np.clip((m[2] - z) / 0.04, 0, 1)
        col = col * (1 - 0.22 * back)[..., None]
        ao = (1 - 0.6 * k_gap[..., 0]) * (1 - 0.5 * k_gl[..., 0]) * (1 - 0.45 * back)
        # relief: perikymata (fine horizontal ridges) + lobes; edges of the incisal band slightly raised
        peri = np.sin(e / 0.00028 * 2 * np.pi + 0.4 * mot) * 0.5e-6
        lobes = _noise(rng, sh, max(0.9e-3 / C.tm, 2.0)) * 2.0e-6
        h = (peri + lobes + 3e-6 * lobe[..., 0]) * msk
        rough = 0.18 + 0.10 * k_cerv[..., 0] + 0.08 * k_gap[..., 0]
        spec = 0.5 - 0.15 * k_gap[..., 0]
        thick = 0.35 + 0.4 * k_inc[..., 0]
        return dict(col=col, ao=ao, rough=rough, spec=spec, thick=thick, height=h)
    return fn


def teeth(texdir, teeth_parts, gum_parts, mouth_lm, n=1024, **kw):
    """teeth_parts / gum_parts: [(name, P, UV, tris, N)] — gum names carry 'upper' / 'lower'."""
    fn = teeth_fn(mouth_lm)
    trees = {}
    for pn, P, UV, T, N in gum_parts:
        trees["upper" if "upper" in pn else "lower"] = cKDTree(_surf(P, T))

    def fn2(C):
        C.gum_trees = trees
        return fn(C)
    return paint("teeth", teeth_parts, fn2, texdir, n=n, typ="teeth", seed=21, **kw)


# ------------------------------------------------------------------ gums
def gums(texdir, gum_parts, teeth_parts, mouth_lm, n=512, **kw):
    m = np.asarray(mouth_lm, float)
    tt = cKDTree(np.concatenate([_surf(p[1], p[3]) for p in teeth_parts]))

    def fn(C):
        sh = (C.n, C.n)
        P, rng = C.P, C.rng
        x, z = P[..., 0], P[..., 2]
        back = np.clip((m[2] - z) / 0.03, 0, 1)
        side = np.clip((np.abs(x) - 0.010) / 0.010, 0, 1)
        k = np.clip(1 - 0.55 * np.maximum(back, side), 0.3, 1)
        dt = np.minimum(tt.query(P.reshape(-1, 3), distance_upper_bound=0.01)[0].reshape(sh), 1.0)
        # healthy attached gingiva: coral pink; the free margin hugging the teeth a touch lighter/pinker,
        # the alveolar mucosa further from the teeth redder and darker (thin, vessels show)
        att = np.array([186, 100, 106]) / 255
        marg = np.array([176, 94, 100]) / 255
        muc = np.array([172, 74, 80]) / 255
        km = _sm(0.0015, 0.0003, dt)[..., None]
        kmu = _sm(0.004, 0.009, dt)[..., None]
        col = att * (1 - kmu) + muc * kmu
        col = col * (1 - 0.5 * km) + marg * 0.5 * km
        # stippling (orange-peel dimples ~0.3-0.5 mm) on the attached gingiva, faint mottling
        sp = max(0.35e-3 / C.tm, 0.7)
        dots = ndimage.gaussian_filter((rng.random(sh) < 0.25 / max(sp, 1) ** 2).astype(float), sp * 0.6)
        dots /= max(dots.max(), 1e-9)
        stip = dots * (1 - kmu[..., 0]) * (1 - 0.7 * km[..., 0])
        mot = _noise(rng, sh, max(1.2e-3 / C.tm, 1.5))
        col = col * (1 - 0.12 * stip[..., None]) * (1 + 0.04 * mot[..., None] * np.array([1.0, 0.6, 0.6]))
        # crevice right at the tooth: a thin dark line
        kc = _sm(0.00035, 0.0, dt)[..., None]
        col = col * (1 - 0.35 * kc)
        col = col * k[..., None]
        h = (-6e-6 * stip + 2e-6 * mot) * C.mask
        rough = 0.30 + 0.08 * stip
        return dict(col=col, ao=k * (1 - 0.4 * kc[..., 0]), rough=rough, height=h)
    return paint("gums", gum_parts, fn, texdir, n=n, typ="gums", seed=22, **kw)


# ------------------------------------------------------------------ tongue
def tongue(texdir, tongue_parts, mouth_lm, n=512, **kw):
    m = np.asarray(mouth_lm, float)

    def fn(C):
        sh = (C.n, C.n)
        P, N, rng = C.P, C.N, C.rng
        x, y, z = P[..., 0], P[..., 1], P[..., 2]
        zmax = P[C.mask][:, 2].max() if C.mask.any() else m[2]
        back = np.clip((m[2] - z) / 0.05, 0, 1)
        tip = _sm(0.012, 0.002, zmax - z)                     # the tip (smoother, pinker)
        dors = _sm(0.1, 0.6, N[..., 1])                         # the upper (dorsal) surface
        # median sulcus: a shallow groove along the midline of the dorsum, fading at the tip and far back
        sw = 0.0011
        groove = np.exp(-(x / sw) ** 2) * dors * (1 - tip) * _sm(0.0, 0.012, zmax - z)
        base = np.array([214, 124, 130]) / 255
        deep = np.array([150, 70, 78]) / 255
        col = base * (1 - 0.5 * back[..., None]) + deep * 0.5 * back[..., None]
        # filiform papillae: a dense velvety coat (fine, slightly whitish-pink) on the dorsum, not the tip/underside
        fp = max(0.22e-3 / C.tm, 0.6)
        fil = _noise(rng, sh, fp)
        filc = np.clip(fil, 0, None) * dors * (1 - 0.6 * tip)
        col = col * (1 + 0.06 * filc[..., None] * np.array([1.0, 1.25, 1.2]))
        # fungiform papillae: sparse red dots ~0.6-1 mm, denser toward the tip and the sides
        fu = max(0.4e-3 / C.tm, 0.8)
        dens = (0.5 + 1.5 * tip + 1.0 * _sm(0.008, 0.016, np.abs(x))) * dors
        dots = (rng.random(sh) < 0.05 * dens / max(fu, 1) ** 2).astype(float)
        dots = ndimage.gaussian_filter(dots, fu * 0.55)
        dots = np.clip(dots / max(np.percentile(dots[dots > 0], 99) if (dots > 0).any() else 1, 1e-6), 0, 1)
        col = col * (1 - 0.28 * dots[..., None]) + np.array([196, 70, 82]) / 255 * 0.28 * dots[..., None]
        col = col * (1 - 0.08 * groove[..., None])
        mot = _noise(rng, sh, max(2e-3 / C.tm, 2))
        col = col * (1 + 0.04 * mot[..., None])
        # underside (ventral): smoother, bluer-red (veins)
        ven = _sm(0.0, -0.5, N[..., 1])[..., None]
        col = col * (1 - 0.3 * ven) + np.array([160, 76, 92]) / 255 * 0.3 * ven
        h = (2.5e-6 * fil * dors * (1 - 0.6 * tip) + 6e-6 * dots - 40e-6 * groove + 3e-6 * mot) * C.mask
        rough = 0.40 + 0.15 * filc - 0.1 * tip
        ao = (1 - 0.4 * back) * (1 - 0.35 * groove)
        return dict(col=col, ao=ao, rough=rough, height=h)
    return paint("tongue", tongue_parts, fn, texdir, n=n, typ="tongue", seed=23, **kw)


# ------------------------------------------------------------------ mouth interior (cheeks, palate, throat)
def mouth_interior(texdir, mouth_parts, mouth_lm, n=512, **kw):
    """v5's colour field kept exactly (dark wet red falling to the buccal sides / throat — SHADE's corner fix relies
    on it); only a faint mucosa mottling, palate rugae ridges and a wet relief on top."""
    m = np.asarray(mouth_lm, float)

    def fn(C):
        sh = (C.n, C.n)
        P, N, rng = C.P, C.N, C.rng
        t = np.maximum(np.clip((m[2] - P[..., 2]) / 0.05, 0, 1), np.clip((np.abs(P[..., 0]) - 0.011) / 0.011, 0, 1))
        front = np.array([196, 112, 110]) / 255
        deep = np.array([140, 66, 68]) / 255
        col = front * (1 - t[..., None]) + deep * t[..., None]
        # falling to near-black only at the very throat / far buccal sides (on top of v5's field)
        far = np.clip((m[2] - P[..., 2] - 0.05) / 0.03, 0, 1)
        col = col * (1 - 0.5 * far[..., None])
        mot = _noise(rng, sh, max(1.5e-3 / C.tm, 1.5))
        ves = np.clip(_noise(rng, sh, max(0.5e-3 / C.tm, 1)), 0, None)
        col = col * (1 + 0.05 * mot[..., None]) * (1 - 0.05 * ves[..., None] * np.array([0.3, 1.0, 0.9]))
        # palate rugae: transverse ridges just behind the upper front teeth (surface facing down)
        pal = _sm(0.2, 0.7, -N[..., 1]) * _sm(0.035, 0.012, m[2] - P[..., 2]) * _sm(0.014, 0.006, np.abs(P[..., 0]))
        rug = np.sin((m[2] - P[..., 2]) / 0.0028 * 2 * np.pi + 3 * np.abs(P[..., 0]) / 0.01) * pal
        h = (8e-6 * rug + 3e-6 * mot) * C.mask
        return dict(col=col, ao=1 - 0.55 * t, rough=np.full(sh, 0.4) - 0.05 * mot.clip(-1, 1), height=h)
    return paint("mouth_interior", mouth_parts, fn, texdir, n=n, typ="mouth_interior", seed=24, **kw)
