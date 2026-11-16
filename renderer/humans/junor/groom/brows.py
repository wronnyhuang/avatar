"""HAIR-BL — realistic brows for a BUILT head (junor.asset vN), derived from the asset alone.

    build(H, rng, log) -> {"brows_l": {...}, "brows_r": {...}}       (regroom.py calls this)

Shape: the brow_l / brow_r landmarks (5 pts, inner -> outer, Multi-PIE) give the centreline on the skin; a visual band
(top / bottom offsets along the skin "up" direction, in mm, as a function of s = 0 head .. 1 tail) is filled with
roots (soft density falloff at every edge and a sparse, feathered head); each hair's direction depends on where it
sits (head: up and a little outward; body: outward along the arch with a herringbone - lower rows rising, upper rows
falling - plus a few cross-hatching hairs; tail: outward-down), its length is clipped so the TIP stays inside the
band (the outline is made by where hairs end, not by a painted edge), and it is laid on the skin (each point
projected onto the head triangles, then lifted 0.1-1.3 mm along the normal: rises off the root, settles back down).
Roots are exact closest points on the head submesh (bind distance ~0) so surface follow rides the brow shapes.

Also exports the mesh helpers lashes.py uses (MeshProj: exact closest point on the head triangles).
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

try:
    from head import nrm, smoothstep, submesh_tris
except ImportError:                                   # imported as a package module
    from .head import nrm, smoothstep, submesh_tris

P_BROW = 9
N_BROW = 1150          # lod0 strands per side (after the density rejection)
N_BROW1 = 560         # lod1
W_BROW = (0.66e-4, 1.5e-5)    # root / tip width (m) - real brow hair is 60-110 um at the root
COL_ROOT = [0.060, 0.032, 0.016]   # a touch lighter / warmer than the scalp root [0.040, 0.021, 0.012]
COL_TIP = [0.095, 0.056, 0.030]

# visual band (mm) about the landmark centreline, along the skin's up direction; s = 0 head .. 1 tail
BAND_S = np.array([0.00, 0.10, 0.30, 0.55, 0.75, 0.90, 1.00])
BAND_TOP = np.array([2.2, 2.9, 2.9, 2.4, 1.5, 0.7, 0.2])
BAND_BOT = np.array([-5.6, -6.1, -5.8, -4.7, -3.2, -1.6, -0.4])
MASK_OVERSAMPLE = 48   # candidates per target strand (mask mode)
MASK_T0, MASK_GAMMA, MASK_ACCEPT = 0.12, 1.1, 1.0   # density -> keep probability
MASK_TIP = 0.10        # a tip must land where the (blurred) mask density exceeds this
HEAD_EXT = 1.5   # mm: the head starts this far medial of landmark 0
TAIL_IN = 4.0    # mm: the tail ends this far medial of landmark 4 (frontal x)
TAIL_DY = -0.8   # mm: tail end height relative to landmark 0


# ------------------------------------------------------------------------------------------------ mesh projection
def closest_on_tris(p, a, b, c):
    """exact closest points of p (n,3) on triangles (a,b,c) (n,3 each) -> (q (n,3), bary (n,3))  [Ericson 5.1.5]"""
    ab, ac, ap = b - a, c - a, p - a
    d1, d2 = (ab * ap).sum(1), (ac * ap).sum(1)
    bp = p - b
    d3, d4 = (ab * bp).sum(1), (ac * bp).sum(1)
    cp = p - c
    d5, d6 = (ab * cp).sum(1), (ac * cp).sum(1)
    va = d3 * d6 - d5 * d4
    vb = d5 * d2 - d1 * d6
    vc = d1 * d4 - d3 * d2
    n = len(p)
    bc = np.zeros((n, 3))
    done = np.zeros(n, bool)

    def put(m, w):
        nonlocal done
        m = m & ~done
        bc[m] = w[m] if np.ndim(w) == 2 else w
        done |= m

    one = np.ones(n)
    z = np.zeros(n)
    put((d1 <= 0) & (d2 <= 0), np.stack([one, z, z], 1))
    put((d3 >= 0) & (d4 <= d3), np.stack([z, one, z], 1))
    v = d1 / np.where(np.abs(d1 - d3) > 1e-30, d1 - d3, 1e-30)
    put((vc <= 0) & (d1 >= 0) & (d3 <= 0), np.stack([1 - v, v, z], 1))
    put((d6 >= 0) & (d5 <= d6), np.stack([z, z, one], 1))
    w = d2 / np.where(np.abs(d2 - d6) > 1e-30, d2 - d6, 1e-30)
    put((vb <= 0) & (d2 >= 0) & (d6 <= 0), np.stack([1 - w, z, w], 1))
    den = (d4 - d3) + (d5 - d6)
    w = (d4 - d3) / np.where(np.abs(den) > 1e-30, den, 1e-30)
    put((va <= 0) & ((d4 - d3) >= 0) & ((d5 - d6) >= 0), np.stack([z, 1 - w, w], 1))
    den = np.where(np.abs(va + vb + vc) > 1e-30, va + vb + vc, 1e-30)
    v, w = vb / den, vc / den
    put(np.ones(n, bool), np.stack([1 - v - w, v, w], 1))
    q = bc[:, 0:1] * a + bc[:, 1:2] * b + bc[:, 2:3] * c
    return q, bc


class MeshProj:
    """closest point on the head submesh (exact, k candidate triangles by centroid) + interpolated vertex normal."""

    def __init__(self, H, names=("head",), materials=("skin_head",), k=16):
        self.V, self.Nv = H.V, H.Nv
        self.T, _ = submesh_tris(H.A, names, materials)
        self.cen = self.V[self.T].mean(1)
        self.kd = cKDTree(self.cen)
        self.k = k

    def project(self, X):
        sh = X.shape
        p = np.asarray(X, float).reshape(-1, 3)
        _, cand = self.kd.query(p, k=self.k, workers=-1)
        bq = np.zeros_like(p)
        bd = np.full(len(p), np.inf)
        bt = np.zeros(len(p), np.int64)
        bb = np.zeros((len(p), 3))
        for j in range(cand.shape[1]):
            t = cand[:, j]
            TT = self.T[t]
            q, bc = closest_on_tris(p, self.V[TT[:, 0]], self.V[TT[:, 1]], self.V[TT[:, 2]])
            d = np.linalg.norm(q - p, axis=1)
            m = d < bd
            bq[m], bd[m], bt[m], bb[m] = q[m], d[m], t[m], bc[m]
        n = nrm((bb[:, :, None] * self.Nv[self.T[bt]]).sum(1))
        self.last_tri = bt.reshape(sh[:-1])            # head-submesh triangle (local index) of each result
        self.last_bary = bb.reshape(sh[:-1] + (3,))
        return bq.reshape(sh), n.reshape(sh), bd.reshape(sh[:-1])


# ------------------------------------------------------------------------------------------------ curve helpers
def catmull(P, n):
    """centripetal-ish Catmull-Rom through P (m,3) with extrapolated ends -> (n,3) dense samples, uniform in t"""
    P = np.asarray(P, float)
    Q = np.vstack([2 * P[0] - P[1], P, 2 * P[-1] - P[-2]])
    out = []
    m = len(P) - 1
    for t in np.linspace(0, 1, n):
        f = t * m
        i = min(int(f), m - 1)
        u = f - i
        p0, p1, p2, p3 = Q[i], Q[i + 1], Q[i + 2], Q[i + 3]
        out.append(0.5 * ((2 * p1) + (-p0 + p2) * u + (2 * p0 - 5 * p1 + 4 * p2 - p3) * u * u
                          + (-p0 + 3 * p1 - 3 * p2 + p3) * u ** 3))
    return np.array(out)


def arc_resample(C, n):
    seg = np.linalg.norm(np.diff(C, axis=0), axis=1)
    L = np.concatenate([[0], np.cumsum(seg)])
    u = np.linspace(0, L[-1], n)
    return np.stack([np.interp(u, L, C[:, k]) for k in range(3)], 1), L[-1]


# ------------------------------------------------------------------------------------------------ brows
def brow_side(H, MP, rng, side, n_target, log=print, mask=None):
    lm = H.A.meta["landmarks"]
    B = np.asarray(lm[f"brow_{side}"], float)[:, :3]
    sg = 1.0 if side == "l" else -1.0
    if sg * (B[-1, 0] - B[0, 0]) < 0:          # make sure inner -> outer
        B = B[::-1]
    # the brow is designed in the FRONTAL plane (x, y) and cast onto the skin along z: the Multi-PIE outer point sits
    # on the temple (following it in 3D wraps a comma-shaped tail round the side of the head). Head = landmark 0
    # (+ a small inward extension), arch from landmarks 1-3, tail end pulled in (TAIL_IN) and up to ~ the head level.
    mm = 1e-3
    ctrl = np.array([
        [B[0, 0] - sg * HEAD_EXT * mm, B[0, 1] - 0.4 * mm],
        [B[1, 0], B[1, 1]],
        [B[2, 0], B[2, 1]],
        [B[3, 0], max(B[3, 1], B[2, 1] - 0.9 * mm)],
        [B[4, 0] - sg * TAIL_IN * mm, B[0, 1] + TAIL_DY * mm]])
    C2 = catmull(np.c_[ctrl, np.zeros(len(ctrl))], 200)
    C2, Ltot = arc_resample(C2, 500)
    C2 = C2[:, :2]
    T2 = nrm(np.gradient(C2, axis=0))
    U2 = np.stack([-T2[:, 1], T2[:, 0]], 1)
    U2[U2[:, 1] < 0] *= -1
    m = len(C2)
    zc = np.interp(np.linspace(0, 1, m), np.linspace(0, 1, 5), B[:, 2])

    def zcast(xy, z0):
        """the skin point straight behind / in front of (x, y): fixed-point of 'project, keep z'"""
        P = np.c_[xy, z0]
        for _ in range(5):
            q, nq, _ = MP.project(P)
            P = np.c_[xy, q[:, 2]]
        return MP.project(P)

    def frame(s):
        i = np.clip(np.rint(s * (m - 1)).astype(int), 0, m - 1)
        return C2[i], T2[i], U2[i], zc[i]

    # irregular outline: low-frequency wobble of both band edges along the brow (+-0.45 mm, a few bumps)
    ph = rng.uniform(0, 2 * np.pi, 6)
    wob = lambda s, k: 0.45e-3 * (0.6 * np.sin(2 * np.pi * 3.1 * s + ph[k]) + 0.4 * np.sin(2 * np.pi * 7.3 * s + ph[k + 3]))
    if mask is not None:
        Th, UVh = MP.T, H.A.mesh["uv0"].astype(np.float64)
        cen3 = MP.cen
        cen2 = cen3[:, :2]
        sgc = sg
        Vh, Nh = MP.V, MP.Nv
        a_ = np.linalg.norm(np.cross(Vh[Th[:, 1]] - Vh[Th[:, 0]], Vh[Th[:, 2]] - Vh[Th[:, 0]]), axis=1)

        def tri_sample_sel(sel, k):
            t = rng.choice(len(sel), k, p=a_[sel] / a_[sel].sum())
            r1, r2 = rng.random(k), rng.random(k)
            q = np.sqrt(r1)
            bc = np.stack([1 - q, q * (1 - r2), q * r2], 1)
            TT = Th[sel][t]
            return ((bc[:, :, None] * Vh[TT]).sum(1), nrm((bc[:, :, None] * Nh[TT]).sum(1)), t, bc)

        def mask_at(uv):
            Hm, Wm = mask.shape
            x = np.clip(uv[:, 0] * Wm - 0.5, 0, Wm - 1.001)
            y = np.clip((1.0 - uv[:, 1]) * Hm - 0.5, 0, Hm - 1.001)
            x0, y0 = x.astype(int), y.astype(int)
            fx, fy = x - x0, y - y0
            return ((mask[y0, x0] * (1 - fx) + mask[y0, x0 + 1] * fx) * (1 - fy)
                    + (mask[y0 + 1, x0] * (1 - fx) + mask[y0 + 1, x0 + 1] * fx) * fy)
    top = lambda s: np.interp(s, BAND_S, BAND_TOP) * 1e-3 + wob(s, 0) * (1 - smoothstep(0.8, 1.0, s))
    bot = lambda s: np.interp(s, BAND_S, BAND_BOT) * 1e-3 + wob(s, 1) * 0.7 * (1 - smoothstep(0.8, 1.0, s))

    kd2 = cKDTree(C2)

    def sv_of(xy):
        _, j = kd2.query(xy)
        return j / (m - 1), ((xy - C2[j]) * U2[j]).sum(1), j

    if mask is not None:
        # ---- roots grown from HEAD's photo-projected brow density (tex/brow_mask.png, head uv atlas): area-uniform
        # candidates on the head skin near the brow, kept with probability ~ density (soft, sparse edges, skin visible
        # between hairs), the inner head feathered; the landmark centreline only orients the direction field
        sel = np.where((np.linalg.norm(cen2[:, None, :] - C2[None, ::10], axis=2).min(1) < 0.012)
                       & (sgc * (cen3[:, 0]) > 0))[0]
        Pc, Nc_, tc, bc = tri_sample_sel(sel, int(n_target * MASK_OVERSAMPLE))
        uvc = (bc[:, :, None] * UVh[Th[sel][tc]]).sum(1)
        dc = mask_at(uvc)
        sc, vc, jc = sv_of(Pc[:, :2])
        okc = (vc > -9e-3) & (vc < 5e-3) & (jc > 0) & (jc < m - 1)
        dc = np.where(okc, dc, 0.0)
        dkd = cKDTree(Pc[:, :2])
        pacc = np.clip((dc - MASK_T0) / (1 - MASK_T0), 0, 1) ** MASK_GAMMA
        pacc *= 0.22 + 0.78 * smoothstep(0.0, 0.26, sc)                  # soft, feathered inner head (HAIR2 r8: sparser - lead 'felt wedge')
        keep = rng.random(len(Pc)) < pacc * MASK_ACCEPT
        roots, nn = Pc[keep], Nc_[keep]
        s, v = sc[keep], vc[keep]
        # band (for the direction field's across-brow coordinate w) = density-weighted percentiles per s bin
        bins = np.linspace(0, 1, 13)
        bi = np.clip(np.digitize(s, bins) - 1, 0, 11)
        lo = np.array([np.percentile(v[bi == k], 8) if (bi == k).sum() > 8 else np.nan for k in range(12)])
        hi = np.array([np.percentile(v[bi == k], 92) if (bi == k).sum() > 8 else np.nan for k in range(12)])
        bc_ = 0.5 * (bins[1:] + bins[:-1])
        g = np.isfinite(lo)
        lo_s, hi_s = np.interp(s, bc_[g], lo[g]), np.interp(s, bc_[g], hi[g])
        w = np.clip((v - lo_s) / np.maximum(hi_s - lo_s, 1e-4), -0.2, 1.2)
        stray = np.zeros(len(s), bool)
        n = len(roots)
        log(f"[brows] {side}: mask candidates {len(Pc)} on-brow {(dc > MASK_T0).sum()} roots {n}")

        def inside_d(xy, tol):
            _, j = dkd.query(xy)
            return dc[j] > MASK_TIP - tol * 60.0        # tip must land on brow density (tolerance in m -> density)
    else:
        # ---- roots: oversample the band, keep with a soft density
        K = int(n_target * 3.2)
        s = rng.random(K)
        w = rng.random(K)                                     # 0 bottom .. 1 top of the band
        tp, bt = top(s), bot(s)
        v = bt + w * (tp - bt)
        edge = 0.12 + 0.88 * smoothstep(0.0, 0.2, w) * (1 - 0.7 * smoothstep(0.85, 1.0, w))
        dens = (edge * (0.6 + 0.4 * smoothstep(0.0, 0.1, s))           # full inner third, only the very head feathers
                * (1 - 0.6 * smoothstep(0.70, 1.0, s)))               # thinning tail
        keep = rng.random(K) < dens
        s, w, v = s[keep], w[keep], v[keep]
        # strays: a few fine hairs scattered just outside the band (below + at the head)
        ns = int(0.045 * n_target)
        ss = rng.random(ns) * 0.85
        below = rng.random(ns) < 0.7
        vs = np.where(below, bot(ss) - rng.exponential(0.9e-3, ns), top(ss) + rng.exponential(0.5e-3, ns))
        s = np.concatenate([s, ss])
        v = np.concatenate([v, vs])
        w = np.concatenate([w, np.where(below, -0.1, 1.1)])
        stray = np.r_[np.zeros(len(s) - ns, bool), np.ones(ns, bool)]
        c2, t2, u2, z0 = frame(s)
        roots, nn, _ = zcast(c2 + u2 * v[:, None], z0)
        n = len(roots)

    c2, t2, u2, z0 = frame(s)
    # ---- direction (angle in the (t, u) brow frame: 0 = outward along the brow, +90 = up)
    head = 1 - smoothstep(0.12, 0.42, s)                  # inner third: hairs brushed up-and-out
    tail = smoothstep(0.62, 1.0, s)
    body_ang = 26.0 * (1 - w) - 14.0 * w                  # herringbone: lower rows rise outward, upper rows fall
    ang = head * (66.0 - 14.0 * w) + (1 - head) * body_ang
    ang = ang * (1 - 0.7 * tail) - 4.0 * tail
    cross = rng.random(n) < 0.12                          # cross-hatching layer
    ang = ang + np.where(cross, rng.choice([-1, 1], n) * rng.uniform(14, 28, n), 0.0)
    ang = ang + rng.normal(0, 5.5 + 6.0 * head, n)
    ang = np.where(stray, rng.uniform(-30, 70, n), ang)
    a = np.radians(ang)
    # the frontal brow frame (t2 outward along the brow, u2 up) carried into the root's tangent plane
    # through the z-cast map's Jacobian (skin z = f(x, y): dz = -(nx dx + ny dy) / nz), so the brow keeps its frontal
    # look on the temple (a plain tangent-plane projection turns the lateral tail hairs downward there)
    d2 = np.cos(a)[:, None] * t2 + np.sin(a)[:, None] * u2
    nz = np.maximum(nn[:, 2], 0.3)
    d = nrm(np.c_[d2, -(nn[:, 0] * d2[:, 0] + nn[:, 1] * d2[:, 1]) / nz])
    d = nrm(d - nn * (d * nn).sum(1, keepdims=True))

    # ---- length (m), clipped so the tip stays inside the band (+ a soft random tolerance)
    L = (5.5 + 3.8 * smoothstep(0.05, 0.4, s) - 3.2 * smoothstep(0.62, 1.0, s)) * 1e-3
    L = L * rng.uniform(0.82, 1.18, n) * (1 - 0.25 * head * rng.random(n))
    L = np.where(stray, rng.uniform(2.5, 5.0, n) * 1e-3, L)
    # clip: the tip (straight-line, frontal) must land inside the band (+ a soft random tolerance) at ITS OWN s,
    # and not past the tail end -> the outline is where hairs end
    tol = (0.15 + rng.exponential(0.45, n)) * 1e-3           # most hairs end at the edge, a few poke past it
    def inside(xy):
        if mask is not None:
            return inside_d(xy, tol)
        sj, vj, j = sv_of(xy)
        past = ((xy - C2[-1]) * T2[-1]).sum(1)
        return (vj <= top(sj) + tol) & (vj >= bot(sj) - tol) & (past <= tol * 0.3) & (j > 0)
    root2 = roots[:, :2] if mask is not None else c2 + u2 * v[:, None]
    fit = np.zeros(n)
    for f in (1.0, 0.88, 0.76, 0.64, 0.52, 0.4):
        okf = (fit == 0) & inside(root2 + d[:, :2] * (L * f)[:, None])
        fit[okf] = f
    L = np.where(stray, L, L * fit)
    ok = L > 2.2e-3
    log(f"[brows] {side}: length fit kept {ok.sum()}/{len(ok)}")
    roots, nn, d, L, s, stray, w = roots[ok], nn[ok], d[ok], L[ok], s[ok], stray[ok], w[ok]
    n = len(roots)
    if n > n_target:
        o = rng.choice(n, n_target, replace=False)
        roots, nn, d, L, s, stray, w = roots[o], nn[o], d[o], L[o], s[o], stray[o], w[o]
        n = n_target

    # ---- the strand: an in-plane arc (gentle bend), conformed to the skin, lifted off and settled back
    t = np.linspace(0, 1, P_BROW)
    bend = np.radians(rng.normal(0, 9, n) - 10 * (1 - smoothstep(0.0, 0.3, s)))   # head hairs curl toward the body
    side_dir = nrm(np.cross(nn, d))
    ang_t = bend[:, None] * t[None]
    # integrate the arc in the tangent plane
    stepdir = (np.cos(ang_t)[..., None] * d[:, None] + np.sin(ang_t)[..., None] * side_dir[:, None])
    Y = roots[:, None] + np.cumsum(stepdir * (L / (P_BROW - 1))[:, None, None], 1) - stepdir[:, :1] * (L / (P_BROW - 1))[:, None, None]
    S_, Ns_, _ = MP.project(Y)
    head = 1 - smoothstep(0.04, 0.30, s)
    lift = rng.uniform(0.8, 1.7, n) * 1e-3 * (0.65 + 0.35 * head)
    h = lift[:, None] * (np.sin(np.pi * np.clip(t, 0, 1) ** 0.75) * 0.85 + 0.15 * t)[None] + 0.12e-3 * (t[None] > 0)
    X = S_ + Ns_ * h[..., None]
    X[:, 0] = roots
    wscale = np.where(stray, 0.6, 1.0) * rng.uniform(0.8, 1.2, n)
    log(f"[brows] {side}: {n} strands, centreline {Ltot * 1000:.1f} mm, L {L.min() * 1e3:.1f}-{L.max() * 1e3:.1f} mm")
    return X, wscale


def _arr(X, wscale, idx, rng, k=1.0):
    S = len(idx)
    ws = wscale[idx] * k
    return {"pos": X[idx].astype(np.float32),
            "width": np.stack([W_BROW[0] * ws, W_BROW[1] * ws], 1).astype(np.float32),
            "seed": rng.random(S).astype(np.float32), "clump": (np.arange(S) + 300000).astype(np.uint32)}


def load_mask(asset_dir, sigma_px=1.2):
    """HEAD's tex/brow_mask.png (photo brow density in the head uv atlas) -> float 0..1, lightly blurred; None if absent"""
    import os
    p = os.path.join(asset_dir, "tex", "brow_mask.png")
    if not os.path.exists(p):
        return None
    from PIL import Image
    import cv2
    m = np.asarray(Image.open(p).convert("L"), np.float32) / 255.0
    return cv2.GaussianBlur(m, (0, 0), sigma_px)


def build(H, rng, log=print):
    MP = MeshProj(H)
    meta = {"bone": "head", "follow": "surface", "surface_submesh": "head", "width_root_m": W_BROW[0],
            "width_tip_m": W_BROW[1], "color_root": COL_ROOT, "color_tip": COL_TIP, "melanin": 0.82,
            "redness": 0.55, "roughness": 0.7, "groom": " soft brows (brow_mask density or landmark band, herringbone)"}
    out = {}
    brng = np.random.default_rng(int(rng.integers(1 << 30)))
    mask = load_mask(H.dir)
    log(f"[brows] {'tex/brow_mask.png density' if mask is not None else 'landmark band (no tex/brow_mask.png)'}")
    for side in ("l", "r"):
        X, ws = brow_side(H, MP, np.random.default_rng(brng.integers(1 << 30)), side, N_BROW, log, mask)
        S = len(X)
        i1 = np.sort(brng.choice(S, min(N_BROW1, S), replace=False))
        out[f"brows_{side}"] = {"meta": dict(meta),
                                "lod0": _arr(X, ws, np.arange(S), brng),
                                "lod1": _arr(X, ws, i1, brng, np.sqrt(S / len(i1)) * 0.85)}
    return out
