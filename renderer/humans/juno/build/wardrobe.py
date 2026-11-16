"""asset — wardrobe: ochre chunky-knit cardigan, cream collared blouse, dark teal trousers, simple flats
(owner: wardrobe helper).

build(base, texdir) -> (parts: [Part], materials: {id: json}, hide_faces: set of MH "body" face ids fully covered)

Everything is derived from the base body (MH vertex groups, dense skin weights, bone heads/tails) so it follows the lead's
design changes. Garment shells = MH body faces -> 1x subdivision -> offset/ease/drape -> smoothing with skin collision ->
static folds; every opening (hem, cuffs, front, neckline, trouser hems, shoe collar) is covered by a SWEPT band with real
thickness, which also hides the jagged cut of the shell. Cloth UVs are in fabric metres / TILE_M[kind] (wrap = repeat).
"""
from __future__ import annotations

import os

import numpy as np
from scipy.spatial import cKDTree

from wardrobe_geo import (BodySDF, QuadMesh, make_part, normalize, rounded_rect, smoothstep, split_by_corner_uv,
                          sweep, top4, tri_normals, weights_from_body)
from parts import Part

try:
    import wardrobe_tex
except Exception:  # noqa: BLE001
    wardrobe_tex = None

SUBDIV = int(os.environ.get('WARDROBE_SUBDIV', '0'))   # 0 keeps lod1 in budget; 1 = smoother shells (4x tris)
TILE_FALLBACK = {"knit": 0.12, "rib": 0.06, "blouse": 0.02, "trousers": 0.03, "leather": 0.05, "buttons": 0.02}


def TILE(kind):
    if wardrobe_tex is not None and hasattr(wardrobe_tex, "TILE_M"):
        return wardrobe_tex.TILE_M[kind]
    return TILE_FALLBACK[kind]


# ----------------------------------------------------------------------------------------------------------------- ctx
class Ctx:
    def __init__(self, base):
        self.base = base
        self.D = base.weights_dense()
        self.sdf = BodySDF(base)
        b = base.bones
        self.bh = {k: np.asarray(v["head"], float) for k, v in b.items()}
        self.bt = {k: np.asarray(v["tail"], float) for k, v in b.items()}
        self.bi = base.bidx
        fids = base.faces_of("body")
        self.body_fids = fids
        self.body_Q = np.array([[vi for vi, _ in base.obj.faces[f]] for f in fids], np.int64)
        V = base.VA
        self.y_thigh = 0.5 * (self.bh["thigh_l"][1] + self.bh["thigh_r"][1])
        self.y_sh = 0.5 * (self.bh["upperarm_l"][1] + self.bh["upperarm_r"][1])
        body = base.verts("body")
        tv = body[(V[body, 1] > self.y_thigh + 0.15) & (V[body, 1] < self.y_sh) & (np.abs(V[body, 0]) < 0.2)]
        self.bust = V[tv[np.argmax(V[tv, 2])]]
        self.zc = float(np.mean([self.bh[k][2] for k in ("pelvis", "spine_01", "spine_02", "spine_03")]))

    def w(self, X, *bones):
        return sum(X[:, self.bi[b]] for b in bones)


def arm_axis(ctx, side):
    """polyline shoulder -> elbow -> wrist and cumulative length (for sleeve params)."""
    s = side
    pts = np.array([ctx.bh[f"upperarm_{s}"], ctx.bh[f"lowerarm_{s}"], ctx.bh[f"hand_{s}"]])
    return pts


def chain_param(X, pts):
    """project points onto a polyline: returns (t = arclength from pts[0], perpendicular vector, segment idx)."""
    best_d = np.full(len(X), np.inf)
    T = np.zeros(len(X))
    PERP = np.zeros_like(X)
    SEG = np.zeros(len(X), np.int64)
    acc = 0.0
    for k in range(len(pts) - 1):
        a, b = pts[k], pts[k + 1]
        L = np.linalg.norm(b - a)
        d = (b - a) / L
        t = np.clip(((X - a) @ d), 0 if k > 0 else -1e9, L if k < len(pts) - 2 else 1e9)
        q = a + t[:, None] * d
        perp = X - q
        dist = np.linalg.norm(perp, axis=1)
        m = dist < best_d
        best_d[m], T[m], PERP[m], SEG[m] = dist[m], acc + t[m], perp[m], k
        acc += L
    return T, PERP, SEG


def surf_tree(P, N):
    return (cKDTree(P), P, N)


def project_to(st, X, lift=0.0):
    tree, P, N = st
    _, i = tree.query(X, k=3)
    q = P[i].mean(1)
    n = normalize(N[i].mean(1))
    Xp = X - n * ((X - q) * n).sum(1, keepdims=True) + n * lift
    return Xp, n


def boundary_loops(Q):
    E = np.concatenate([Q[:, [0, 1]], Q[:, [1, 2]], Q[:, [2, 3]], Q[:, [3, 0]]])
    key = np.sort(E, 1)
    u, inv, c = np.unique(key, axis=0, return_inverse=True, return_counts=True)
    bd = E[c[inv] == 1]
    nxt = {int(a): int(b) for a, b in bd}
    loops, seen = [], set()
    for s in nxt:
        if s in seen:
            continue
        loop, v = [], s
        while v not in seen and v in nxt:
            seen.add(v)
            loop.append(v)
            v = nxt[v]
        loops.append(np.array(loop))
    return loops


def smooth_chain(C, iters=30, closed=False, lam=0.5):
    C = C.copy()
    for _ in range(iters):
        if closed:
            L = 0.5 * (np.roll(C, 1, 0) + np.roll(C, -1, 0)) - C
        else:
            L = np.zeros_like(C)
            L[1:-1] = 0.5 * (C[:-2] + C[2:]) - C[1:-1]
        C += lam * L
    return C


def resample(C, step, closed=False):
    if closed:
        C = np.vstack([C, C[:1]])
    s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(C, axis=0), axis=1))]
    n = max(int(s[-1] / step), 4)
    t = np.linspace(0, s[-1], n + (0 if closed else 1), endpoint=not closed)
    return np.stack([np.interp(t, s, C[:, k]) for k in range(3)], 1)


def band_profile(a, b, th, lift, r=None, nr=1):
    """rounded rectangle across [a, b] (metres, + = Bn side), up [lift, lift + th]."""
    r = min(th * 0.45, abs(b - a) * 0.3) if r is None else r
    pr = rounded_rect(abs(b - a), th, r, nr=nr)
    pr[:, 0] += 0.5 * (a + b)
    pr[:, 1] += lift
    return pr


def dense_from(P_src, D_src, X, k=4):
    d, i = cKDTree(P_src).query(X, k=k)
    w = 1.0 / np.maximum(d, 1e-4) ** 2
    w /= w.sum(1, keepdims=True)
    return (D_src[i] * w[:, :, None]).sum(1)


def sweep_part_arrays(C, N, prof, closed, tile, D_src_P, D_src, cap=True):
    P, Q, UV, weld, capT = sweep(C, N, prof, closed=closed, cap_ends=cap)
    tris = np.concatenate([Q[:, [0, 1, 2]], Q[:, [0, 2, 3]]] + ([capT] if len(capT) else []))
    D = dense_from(D_src_P, D_src, P)
    return P, tris, UV / tile, D, weld


class Acc:
    """accumulates several pieces into one Part."""

    def __init__(self):
        self.P, self.T, self.UV, self.D, self.weld = [], [], [], [], []
        self.n = 0
        self.wn = 0

    def add(self, P, T, UV, D, weld=None):
        weld = np.arange(len(P)) if weld is None else np.asarray(weld)
        self.P.append(P); self.T.append(np.asarray(T) + self.n); self.UV.append(UV); self.D.append(D)
        self.weld.append(weld + self.wn)
        self.n += len(P)
        self.wn += int(weld.max()) + 1

    def part(self, name, mat, double_sided=False):
        D = np.concatenate(self.D)
        return make_part(name, mat, np.concatenate(self.P), np.concatenate(self.T), np.concatenate(self.UV), D,
                         weld=np.concatenate(self.weld), double_sided=double_sided)


# ------------------------------------------------------------------------------------------------------------ cardigan
def regions(ctx, B, Dm):
    """0 torso, 1 sleeve_l, 2 sleeve_r (by arm skin weights); sleeve_t = distance from the wrist up the arm."""
    region = np.zeros(len(B), int)
    sleeve_t = np.zeros(len(B))
    for k, s in ((1, "l"), (2, "r")):
        aw = ctx.w(Dm, f"upperarm_{s}", f"lowerarm_{s}", f"hand_{s}")
        sl = aw > 0.5
        pts = arm_axis(ctx, s)
        region[sl] = k
        t, _, _ = chain_param(B, pts)
        L = np.linalg.norm(pts[1] - pts[0]) + np.linalg.norm(pts[2] - pts[1])
        sleeve_t[sl] = (L - t)[sl]
    return region, sleeve_t

def cardigan(ctx):
    base, D = ctx.base, ctx.D
    V = base.VA
    Qb = ctx.body_Q
    cen = V[Qb].mean(1)
    Dc = D[Qb].mean(1)
    hand = sum(Dc[:, ctx.bi[b]] for b in ctx.bi if b.startswith(("hand", "thumb", "index", "middle", "ring", "pinky")))
    headn = ctx.w(Dc, "head", "neck_01")
    legs = ctx.w(Dc, "thigh_l", "thigh_r", "calf_l", "calf_r", "foot_l", "foot_r")
    hem_bottom = ctx.y_thigh - 0.045               # hip length
    rib_w = 0.055
    y_cut = hem_bottom + rib_w - 0.012
    sel = (cen[:, 1] > y_cut - 0.04) & (hand < 0.3) & (headn < 0.85) & (legs < 0.7)
    cids = ctx.body_fids[sel]
    m = QuadMesh.from_body(base, cids, D)
    for _ in range(SUBDIV):
        m = m.subdivide()
    B = m.B
    # ---- classify vertices
    neck_w = ctx.w(m.D, "neck_01", "head")
    y_v = hem_bottom + 0.60 * (ctx.y_sh - hem_bottom)     # V point / top button
    neck_c = ctx.bh["neck_01"]
    x_snp = 0.055 + 0.0 * neck_c[0]
    y_snp = neck_c[1] - 0.01
    front = B[:, 2] > ctx.zc + 0.02
    wv = np.where(B[:, 1] < y_v, 0.0, (B[:, 1] - y_v) / (y_snp - y_v) * x_snp)
    wv = np.where(B[:, 1] > y_snp, 1.0, wv)
    open_v = front & (np.abs(B[:, 0]) < wv)
    region, sleeve_t = regions(ctx, B, m.D)
    cuff_w = 0.06
    wrist_cut = cuff_w - 0.014
    keep_v = np.where(region == 0, (B[:, 1] > y_cut) & (neck_w < 0.3) & ~open_v, sleeve_t > wrist_cut)
    fk = keep_v[m.Q].sum(1) >= 3
    cov = set(int(x) for x in cids[keep_v[m.Q].all(1).reshape(-1, 4 ** SUBDIV).all(1)])
    m = m.subset(fk)
    # recompute per-vertex attrs on the subset
    B = m.B
    region, sleeve_t = regions(ctx, B, m.D)
    Nb = quad_normals_safe(B, m.Q)
    # ---- offset (ease)
    neck_w = ctx.w(m.D, "neck_01", "head")
    off = np.where(region == 0, 0.018, 0.018) - 0.006 * smoothstep(0.0, 0.3, neck_w)   # ASSET3 v7: a little oversized
    P = B + Nb * off[:, None]
    # drape envelope on the torso below the bust: cloth hangs from bust / shoulder blades
    tor = region == 0
    y_b = ctx.bust[1]
    phi = np.arctan2(P[:, 0], P[:, 2] - ctx.zc)
    r = np.hypot(P[:, 0], P[:, 2] - ctx.zc)
    nb, ny = 96, 200
    y0, y1 = P[tor, 1].min() - 0.01, y_b + 0.02
    bi_ = np.clip(((phi + np.pi) / (2 * np.pi) * nb).astype(int), 0, nb - 1)
    yi = np.clip(((P[:, 1] - y0) / (y1 - y0) * ny).astype(int), 0, ny - 1)
    G = np.zeros((nb, ny))
    msk = tor & (P[:, 1] < y1) & (ctx.w(m.D, 'upperarm_l', 'upperarm_r', 'lowerarm_l', 'lowerarm_r', 'clavicle_l', 'clavicle_r') < 0.05)
    np.maximum.at(G, (bi_[msk], yi[msk]), r[msk])
    for j in range(ny - 2, -1, -1):
        G[:, j] = np.maximum(G[:, j], G[:, j + 1] - 0.03 * (y1 - y0) / ny)
    from scipy.ndimage import uniform_filter1d
    G = uniform_filter1d(G, 5, axis=0, mode="wrap")
    if BOXY_FRONT:   # ASSET3 v7: a flat (boxy) front -- the knit bridges straight across the chest, no shaped bust
        phc = -np.pi + (np.arange(nb) + 0.5) * 2 * np.pi / nb
        sect = np.abs(phc) < np.radians(58)
        cz = np.cos(phc)
        zpl = (G[sect] * cz[sect, None]).max(0)                     # per y row: the most forward point of the front
        flat = np.where(sect[:, None], zpl[None] / np.maximum(cz[:, None], 0.3), 0)
        wsec = np.clip((np.radians(70) - np.abs(phc)) / np.radians(14), 0, 1)[:, None]
        G = np.maximum(G, G + (flat - G) * wsec * (flat > G))
        G = uniform_filter1d(G, 3, axis=0, mode="wrap")
    env = G[bi_, yi]
    grow = np.where(tor & (P[:, 1] < y1), np.clip(env - r, 0, 0.05) * 0.95, 0)
    # do not bulge the side seams into the arms' space
    dirr = np.stack([np.sin(phi), np.zeros_like(phi), np.cos(phi)], 1)
    P = P + dirr * grow[:, None]
    # ---- smoothing + collision rounds
    pin = None
    for it in range(5):
        P = m.smooth(P, iters=6, pin=pin)
        P = ctx.sdf.push_out(P, off * 0.85)
    # bust: iron out the nipple bumps (cloth bridges them)
    bm = np.zeros(len(P))
    for sx in (1, -1):
        c = np.array([sx * max(abs(ctx.bust[0]), 0.06), ctx.bust[1], ctx.bust[2]])   # ASSET3: flat proxy -> bust x may be 0
        bm = np.maximum(bm, np.exp(-(np.linalg.norm(P - c, axis=1) / 0.065) ** 2))
    P2 = m.smooth(P, iters=40)
    P = P + (P2 - P) * bm[:, None]
    # ---- folds (static, along the normal)
    N = quad_normals_safe(P, m.Q)
    P = P + N * cardigan_folds(ctx, m, P, region, sleeve_t)[:, None]
    P = ctx.sdf.push_out(P, 0.006)
    if POCKETS:   # ASSET3 v7: the pocket patch sits proud of the front panel
        info_ = dict(hem_bottom=hem_bottom, rib_w=rib_w)
        Np = quad_normals_safe(P, m.Q)
        for x0, y0, pw, ph in pocket_rects(info_):
            dx = np.clip((0.5 * pw - np.abs(P[:, 0] - x0)) / 0.01, 0, 1)
            dy = np.clip((0.5 * ph - np.abs(P[:, 1] - y0)) / 0.01, 0, 1)
            mk = dx * dy * ((region == 0) & (P[:, 2] > ctx.zc + 0.03))
            P = P + Np * (0.0022 * mk)[:, None]
    m.P = P
    m.region, m.sleeve_t = region, sleeve_t
    # smooth weights over the garment (stable cloth skinning)
    Dg = m.lap_smooth(m.D, iters=12)
    m.D = Dg
    return m, dict(covered=cov, hem_bottom=hem_bottom, y_cut=y_cut, y_v=y_v, rib_w=rib_w, cuff_w=cuff_w)


def quad_normals_safe(P, Q):
    from wardrobe_geo import quad_normals
    return quad_normals(P, Q)


def noise3(X, freq, seed=0):
    """cheap smooth noise: sum of a few random-direction sinusoids."""
    rng = np.random.default_rng(seed)
    out = np.zeros(len(X))
    for _ in range(6):
        d = normalize(rng.normal(size=3))
        ph = rng.uniform(0, 2 * np.pi)
        out += np.sin(X @ d * freq * 2 * np.pi + ph)
    return out / 6 ** 0.5


def cardigan_folds(ctx, m, P, region, sleeve_t):
    disp = np.zeros(len(P))
    B = m.B
    for k, s in ((1, "l"), (2, "r")):
        sl = region == k
        pts = arm_axis(ctx, s)
        Lu = np.linalg.norm(pts[1] - pts[0])
        Lf = np.linalg.norm(pts[2] - pts[1])
        t = sleeve_t[sl]
        # crook of the elbow: inner side (toward the forearm's bend direction)
        ua, fa = normalize(pts[1] - pts[0]), normalize(pts[2] - pts[1])
        bend = normalize(fa - ua * (fa @ ua) + np.array([0, 0, 1e-3]))
        rad = normalize(B[sl] - pts[1] - ua * ((B[sl] - pts[1]) @ ua)[:, None])
        inner = np.clip(rad @ bend, 0, 1) ** 1.5
        de = np.abs(t - Lf)
        crook = np.exp(-(de / 0.05) ** 2) * inner
        w = noise3(B[sl], 4.0, seed=k)
        disp[sl] += 0.0068 * crook * np.sin((t - Lf) * 2 * np.pi / 0.028 + w * 1.5)
        # bunching above the cuff (rings, broken up by noise)
        bunch = smoothstep(0.04, 0.06, t) * (1 - smoothstep(0.10, 0.17, t))
        disp[sl] += 0.0035 * bunch * np.sin(t * 2 * np.pi / 0.024 + 2.0 * noise3(B[sl], 6.0, seed=10 + k))
        # gentle long folds along the upper arm
        disp[sl] += 0.0015 * noise3(B[sl] * np.array([1, 1, 1]), 8.0, seed=20 + k)
        # ASSET2 v6: armpit drape — diagonal folds radiating from the underarm onto the upper sleeve
        ax0 = pts[0]
        dv = B[sl] - ax0
        r_ = np.linalg.norm(dv, axis=1)
        under = np.clip(-(dv @ np.array([0, 1.0, 0])) / (r_ + 1e-9), 0, 1)
        ang_ = np.arctan2(dv[:, 2], np.abs(dv[:, 0]) + 1e-9)
        disp[sl] += 0.004 * np.exp(-((r_ - 0.05) / 0.035) ** 2) * under * np.sin(ang_ * 7.0 + noise3(B[sl], 5.0, seed=30 + k))
    tor = region == 0
    y = B[tor, 1]
    side = np.clip(np.abs(B[tor, 0]) / 0.16, 0, 1) ** 2
    waist_y = 0.5 * (ctx.y_thigh + ctx.bust[1])
    wz = np.exp(-((y - waist_y) / 0.07) ** 2)
    disp[tor] += 0.003 * wz * side * np.sin(y * 2 * np.pi / 0.035 + 1.2 * noise3(B[tor], 5.0, seed=3))
    # soft vertical drape on the fronts below the bust
    lowfront = (B[tor, 2] > ctx.zc + 0.04) * (1 - smoothstep(ctx.bust[1] - 0.12, ctx.bust[1] - 0.02, y))
    disp[tor] += 0.0035 * lowfront * np.sin(B[tor, 0] * 2 * np.pi / 0.075 + 1.5 * noise3(B[tor], 3.0, seed=4))
    disp[tor] += 0.0012 * noise3(B[tor], 10.0, seed=5)
    return disp


POCKETS = True
BOXY_FRONT = False   # envelope variant (b2: made a ledge) -- CHEST_POLY instead
CHEST_POLY = True


def pocket_rects(info):
    """(x centre, y centre, width, height) of the two patch pockets on the lower fronts."""
    yb = info["hem_bottom"] + info["rib_w"] + 0.055
    return [(sx * 0.092, yb, 0.105, 0.115) for sx in (1.0, -1.0)]


def cardigan_parts(ctx, m, info):
    """shell part (panel UVs, seams at sleeve underarm and armhole) + rib bands + buttons."""
    P, Q, B = m.P, m.Q, m.B
    region = m.region
    tile = TILE("knit")
    # ---- per-corner UVs
    fr = np.array([np.bincount(region[q], minlength=3).argmax() for q in Q])
    cornerUV = np.zeros(Q.shape + (2,))
    Rt = 0.15
    tf = fr == 0
    phi = np.arctan2(B[:, 0], -(B[:, 2] - ctx.zc))  # 0 = back centre, +-pi = front opening
    cornerUV[tf, :, 0] = phi[Q[tf]] * Rt
    cornerUV[tf, :, 1] = B[Q[tf], 1]
    for k, s in ((1, "l"), (2, "r")):
        ff = fr == k
        pts = arm_axis(ctx, s)
        a = normalize(pts[0] - pts[2])
        g = np.array([0, -1.0, 0]) - a * (-a[1])
        b1 = normalize(g)
        b2 = np.cross(a, b1)
        X = B[Q[ff]].reshape(-1, 3)
        rel = X - pts[1]
        ang = np.arctan2(rel @ b2, rel @ b1)  # 0 = underside (seam)
        ang = np.where(ang < 0, ang + 2 * np.pi, ang).reshape(-1, 4)
        # per-face unwrap
        ref = ang[:, :1]
        ang = np.where(ang - ref > np.pi, ang - 2 * np.pi, np.where(ref - ang > np.pi, ang + 2 * np.pi, ang))
        cornerUV[ff, :, 0] = ang * 0.055
        cornerUV[ff, :, 1] = m.sleeve_t[Q[ff]]
    Pl, tris, UV, Dl, weld, _ = split_by_corner_uv(P, Q, cornerUV / tile, m.D)
    acc_knit = Acc()
    acc_knit.add(Pl, tris, UV, Dl, weld)
    # ---- surface for bands
    Ng = quad_normals_safe(P, Q)
    st = surf_tree(P, Ng)
    acc_rib = Acc()
    rtile = TILE("rib")
    th = 0.0075
    loops = boundary_loops(Q)
    buttons = None
    for lp in loops:
        C = P[lp]
        Bl = B[lp]
        if region[lp].max() > 0 and (region[lp] > 0).mean() > 0.8:      # cuff
            Cs = resample(smooth_chain(C, 20, closed=True), 0.0100, closed=True)
            Cs, Ns = project_to(st, Cs)
            k = int(np.bincount(region[lp]).argmax())
            pts = arm_axis(ctx, "l" if k == 1 else "r")
            a = normalize(pts[2] - pts[1])
            Tn = normalize(np.roll(Cs, -1, 0) - np.roll(Cs, 1, 0))
            Bn = np.cross(Ns, Tn)
            sg = np.sign(np.mean(Bn @ a))  # + = toward the hand
            prof = band_profile(sg * -0.014 if sg > 0 else -0.046, sg * 0.046 if sg > 0 else 0.014, th, -0.003)
            Pp, tri, uv, Dd, wd = sweep_part_arrays(Cs, Ns, prof, True, rtile, P, m.D)
            # taper the cuff toward the wrist (rib pulls in)
            along = (Pp - pts[2]) @ a
            fr_ = smoothstep(-0.03, 0.0, along)
            radial = Pp - pts[2] - a * along[:, None]
            Pp = Pp - normalize(radial) * (0.010 * fr_)[:, None]
            Pp = ctx.sdf.push_out(Pp, 0.003)
            acc_rib.add(Pp, tri, uv, Dd, wd)
            continue
        rw = info["rib_w"]
        if C[:, 1].mean() < info["y_cut"] + 0.03:        # ---- hem: closed loop, band hangs below the cut
            Hs = resample(smooth_chain(C, 20, closed=True), 0.0100, closed=True)
            Hs, Hn = project_to(st, Hs)
            Tn = normalize(np.roll(Hs, -1, 0) - np.roll(Hs, 1, 0))
            Bn = np.cross(Hn, Tn)
            sg = np.sign(np.mean(Bn[:, 1]))
            a0, a1 = (-(rw - 0.012), 0.012) if sg > 0 else (-0.012, rw - 0.012)
            prof = band_profile(a0, a1, th, -0.003)
            Pp, tri, uv, Dd, wd = sweep_part_arrays(Hs, Hn, prof, True, rtile, P, m.D)
            Pp = ctx.sdf.push_out(Pp, 0.008)
            acc_rib.add(Pp, tri, uv, Dd, wd)
            continue
        # ---- neckline loop (V + collar edge): band on the shell side
        Cs = resample(smooth_chain(C, 12, closed=True), 0.0100, closed=True)
        Cs, Ns = project_to(st, Cs)
        Tn = normalize(np.roll(Cs, -1, 0) - np.roll(Cs, 1, 0))
        Bn = np.cross(Ns, Tn)
        cen_ = Cs.mean(0)
        away = np.sign(np.mean(((Cs - cen_) * Bn).sum(1)))      # + = Bn points away from the neck centre = shell side
        bw = 0.026
        a0, a1 = (-0.010, bw) if away > 0 else (-bw, 0.010)
        prof = band_profile(a0, a1, th, -0.003)
        Pp, tri, uv, Dd, wd = sweep_part_arrays(Cs, Ns, prof, True, rtile, P, m.D)
        Pp = ctx.sdf.push_out(Pp, 0.005)
        acc_rib.add(Pp, tri, uv[:, ::-1], Dd, wd)
    # ---- centre-front button band, hem -> V point (overlapping panel), + buttons
    ys = np.arange(info["hem_bottom"] - 0.0, info["y_v"] + 0.004, 0.007)
    ray_y = ys
    # front surface point at x = 0: nearest shell vertices with big z
    cand = np.where((np.abs(P[:, 0]) < 0.02) & (P[:, 2] > ctx.zc))[0]
    Fs = []
    for yy in ray_y:
        j = cand[np.argsort(np.abs(P[cand, 1] - yy))[:6]]
        Fs.append([0.0, yy, P[j, 2].mean()])
    Fs = np.array(Fs)
    Fs, Fn = project_to(st, Fs)
    bwid = 0.030
    prof = band_profile(-bwid / 2, bwid / 2, th, -0.002)
    Pp, Qq, uv, wd, capT = sweep(Fs, Fn, prof, closed=False, cap_ends=True)
    Pp = ctx.sdf.push_out(Pp, 0.005)
    tri = np.concatenate([Qq[:, [0, 1, 2]], Qq[:, [0, 2, 3]], capT])
    acc_rib.add(Pp, tri, (uv / rtile)[:, ::-1], dense_from(P, m.D, Pp), wd)
    # ---- ASSET3 v7: patch pockets (judge r4: blank knit fronts): a rib-edged outline + a ribbed top welt on each front
    if POCKETS:
        candf = np.where((P[:, 2] > ctx.zc + 0.03) & (region == 0))[0]
        kd2 = cKDTree(P[candf][:, :2])
        def onfront(xy):
            _, j = kd2.query(xy, k=4)
            z = P[candf][j][:, :, 2].mean(1)
            return project_to(st, np.c_[xy, z])
        for x0, y0, pw, ph in pocket_rects(info):
            r_ = 0.012
            th_ = np.linspace(0, 2 * np.pi, 72, endpoint=False)
            # rounded-rect outline: superellipse
            ex = 0.5 * pw * np.sign(np.cos(th_)) * np.abs(np.cos(th_)) ** 0.25
            ey = 0.5 * ph * np.sign(np.sin(th_)) * np.abs(np.sin(th_)) ** 0.25
            Cs, Ns = onfront(np.c_[x0 + ex, y0 + ey])
            Cs = Cs + Ns * 0.0025
            Cs = resample(smooth_chain(Cs, 2, closed=True), 0.011, closed=True)
            Cs, Ns = project_to(st, Cs, lift=0.0025)
            prof = band_profile(-0.0035, 0.0035, 0.0032, -0.0012)
            Pp, tri, uv, Dd, wd = sweep_part_arrays(Cs, Ns, prof, True, rtile, P, m.D)
            acc_rib.add(Pp, tri, uv, Dd, wd)
            xs = np.linspace(x0 - 0.5 * pw + 0.006, x0 + 0.5 * pw - 0.006, 12)
            Ct, Nt = onfront(np.c_[xs, np.full(len(xs), y0 + 0.5 * ph - 0.010)])
            Ct, Nt = project_to(st, Ct, lift=0.003)
            prof = band_profile(-0.009, 0.009, 0.0035, -0.001)
            Pp, Qq, uv, wd, capT = sweep(Ct, Nt, prof, closed=False, cap_ends=True)
            tri = np.concatenate([Qq[:, [0, 1, 2]], Qq[:, [0, 2, 3]], capT])
            acc_rib.add(Pp, tri, (uv / rtile)[:, ::-1], dense_from(P, m.D, Pp), wd)
    nbtn = 5
    by = np.linspace(info["hem_bottom"] + 0.035, info["y_v"] - 0.012, nbtn)
    bp, bn = [], []
    for yy in by:
        i = np.argmin(np.abs(Fs[:, 1] - yy))
        bp.append(Fs[i] + Fn[i] * (th - 0.002))
        bn.append(Fn[i])
    buttons = (np.array(bp), np.array(bn))
    return acc_knit, acc_rib, buttons


def button_mesh(ctx, centers, normals, P_src, D_src, radius=0.0095, thick=0.0042, seg=12):
    """lathe a domed button with a rim and a dished centre + 4 holes (as dimples) per centre."""
    prof = np.array([[0.0, thick * 0.55], [0.35, thick * 0.6], [0.55, thick * 0.78], [0.75, thick * 0.98],
                     [0.9, thick * 1.0], [0.98, thick * 0.85], [1.0, thick * 0.45], [0.96, 0.0], [0.0, 0.0]])
    acc = Acc()
    for c, n in zip(centers, normals):
        n = normalize(n)
        t1 = normalize(np.cross(n, [0, 1.0, 0]) if abs(n[1]) < 0.9 else np.cross(n, [1.0, 0, 0]))
        t2 = np.cross(n, t1)
        a = np.linspace(0, 2 * np.pi, seg, endpoint=False)
        P, UV = [], []
        for rr, hh in prof:
            for ang in a:
                d = np.cos(ang) * t1 + np.sin(ang) * t2
                P.append(c + d * rr * radius + n * hh)
                UV.append((0.5 + 0.5 * rr * np.cos(ang), 0.5 + 0.5 * rr * np.sin(ang)))
        P = np.array(P)
        nr = len(prof)
        T = []
        for i in range(nr - 1):
            for j in range(seg):
                a0, a1 = i * seg + j, i * seg + (j + 1) % seg
                b0, b1 = (i + 1) * seg + j, (i + 1) * seg + (j + 1) % seg
                T += [(a0, b0, b1), (a0, b1, a1)]
        T = np.array(T)
        # orientation: outward = away from the centre axis on the dome side; fix with a centroid test
        fn = np.cross(P[T[:, 1]] - P[T[:, 0]], P[T[:, 2]] - P[T[:, 0]])
        cc = c + n * thick * 0.4
        if (((P[T].mean(1) - cc) * fn).sum(1) < 0).mean() > 0.5:
            T = T[:, ::-1]
        # holes: 4 dimples near the centre
        for hx, hy in ((1, 1), (-1, 1), (-1, -1), (1, -1)):
            hc = c + (t1 * hx + t2 * hy) * radius * 0.2 + n * thick * 0.55
            d = np.linalg.norm(P - hc, axis=1)
            P = P - n * (0.0014 * np.exp(-(d / (radius * 0.09)) ** 2))[:, None]
        D = dense_from(P_src, D_src, P)
        acc.add(P, T, np.array(UV), D)
    return acc


# ----------------------------------------------------------------------------------------------------------- materials
def flat_materials(texdir):
    from PIL import Image
    cols = {"cloth_knit": (196, 142, 52), "cloth_rib": (190, 136, 48), "cloth_blouse": (238, 228, 206),
            "cloth_trousers": (34, 78, 82), "shoes": (82, 38, 30), "cloth_buttons": (70, 45, 30)}
    mats = {}
    for mid, c in cols.items():
        for suf in ("", "_1k"):
            Image.fromarray(np.tile(np.array([*c], np.uint8), (16, 16, 1))).save(os.path.join(texdir, f"{mid}_albedo{suf}.png"))
            Image.fromarray(np.tile(np.array([128, 128, 255], np.uint8), (16, 16, 1))).save(os.path.join(texdir, f"{mid}_normal{suf}.png"))
            Image.fromarray(np.tile(np.array([255, 230, 128, 30], np.uint8), (16, 16, 1))).save(os.path.join(texdir, f"{mid}_packed{suf}.png"))
        mats[mid] = {"type": "cloth", "albedo": f"tex/{mid}_albedo.png", "normal": f"tex/{mid}_normal.png",
                     "packed": f"tex/{mid}_packed.png", "params": {"sheen_color": [0.9, 0.8, 0.6], "sheen_roughness": 0.6}}
    return mats


# --------------------------------------------------------------------------------------------------------------- build
CHEST_PROXY_ITERS = 320   # ASSET2 v6: cloth is built on a smoothed, flattened chest (no bust apex detail through the knit)


CHEST_PROXY_MAX_M = 0.07   # ASSET3 v7: boxy straight drape (judge r4: shaped bust read through the knit)


def chest_proxy(base):
    """body positions with the front chest relaxed (plain Laplacian = shrinks the bust apex, fills the cleavage a bit).
    Only the cloth is built on it; the skin under the cardigan/blouse is hidden anyway."""
    from juno_base import tri_of, vertex_normals
    import scipy.sparse as sp
    V = base.VA.copy()
    D = base.weights_dense()
    bi = base.bidx
    m = np.clip(D[:, bi["spine_03"]] + D[:, bi["spine_02"]] + 0.5 * D[:, bi["spine_01"]], 0, 1)
    zc = float(np.mean([base.bones[k]["head"][2] for k in ("spine_01", "spine_02", "spine_03")]))
    m *= np.clip((V[:, 2] - zc) / 0.04, 0, 1)                 # front only
    m *= 1 - np.clip((np.abs(V[:, 0]) - 0.12) / 0.04, 0, 1)    # not the armpits/sides
    if CHEST_POLY:   # ASSET3: the whole front chest (bone weights left the bust apex partly unmasked)
        sp3 = base.bones["spine_03"]["head"]
        nk = base.bones["neck_01"]["head"]
        reg = np.clip((V[:, 2] - zc) / 0.03, 0, 1) * (1 - np.clip((np.abs(V[:, 0]) - 0.13) / 0.03, 0, 1)) \
            * np.clip((nk[1] - 0.04 - V[:, 1]) / 0.04, 0, 1) * np.clip((V[:, 1] - (sp3[1] - 0.22)) / 0.05, 0, 1)
        m = np.maximum(m, reg)
    m[~base.skin_mask] = 0
    tris = np.array([t for fi in base.obj.groups["body"] for t in tri_of([vi for vi, _ in base.obj.faces[fi]])])
    n = len(V)
    e = np.concatenate([tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]])
    e = np.concatenate([e, e[:, ::-1]])
    A = sp.coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(n, n)).tocsr()
    A.data[:] = 1.0
    deg = np.maximum(np.asarray(A.sum(1)).ravel(), 1)
    for _ in range(0 if CHEST_POLY else CHEST_PROXY_ITERS):
        V[:, 2] = V[:, 2] + 0.5 * m * (A @ V[:, 2] / deg - V[:, 2])   # depth only: x/y (silhouette, seams) stay
    if CHEST_POLY:
        # ASSET3 v7: boxy chest -- the front torso depth is a smooth low-order surface z(x, y) fitted on the chest WITHOUT
        # the bust (cylinder-like front); the knit is built on it, so no shaped bust reads through
        sel = np.where(m > 0.05)[0]
        X0 = base.VA[sel]
        sp = np.mean([base.bones[k]["head"] for k in ("spine_02", "spine_03")], 0)
        apex = []
        for sg in (1.0, -1.0):
            c = sel[(sg * X0[:, 0] > 0.03) & (X0[:, 1] > sp[1] - 0.12)]
            apex.append(base.VA[c[np.argmax(base.VA[c, 2])]])
        far = np.min([np.linalg.norm(X0 - a, axis=1) for a in apex], 0) > 0.085
        # front depth as a height field z(x, y) on a 5 mm grid, blurred with a 4.5 cm gaussian: the bust becomes a soft
        # broad swell of the whole front (no apex, no cleavage), like a knit hanging over a cylinder-ish torso
        from scipy.interpolate import griddata
        from scipy.ndimage import gaussian_filter, map_coordinates
        h = 0.005
        xs = np.arange(-0.20, 0.2001, h)
        ys = np.arange(X0[:, 1].min() - 0.05, X0[:, 1].max() + 0.05, h)
        GX, GY = np.meshgrid(xs, ys, indexing="ij")
        allf = np.where((base.VA[:, 2] > zc) & base.skin_mask)[0]
        Z = griddata(base.VA[allf, :2], base.VA[allf, 2], (GX, GY), method="linear")
        Zn = griddata(base.VA[allf, :2], base.VA[allf, 2], (GX, GY), method="nearest")
        Z = np.where(np.isfinite(Z), Z, Zn)
        Zb = gaussian_filter(Z, 9.0, mode="nearest")
        zfit = map_coordinates(Zb, [(X0[:, 0] - xs[0]) / h, (X0[:, 1] - ys[0]) / h], order=1, mode="nearest")
        zfit = np.minimum(zfit, X0[:, 2] + 0.015)          # bridge the cleavage, never bulge the sides out
        print("chest blur: max in mm", float((X0[:, 2] - zfit).max() * 1e3), "max out", float((zfit - X0[:, 2]).max() * 1e3))
        V[sel, 2] = V[sel, 2] + m[sel] * (zfit - V[sel, 2])
    V[:, 2] = np.maximum(V[:, 2], base.VA[:, 2] - CHEST_PROXY_MAX_M)      # flatten the apex, never collapse the torso
    return V, vertex_normals(V, base.tris)


def build(base, texdir):
    V0, N0 = base.VA, base.N
    if CHEST_PROXY_ITERS:
        base.VA, base.N = chest_proxy(base)
    try:
        return _build(base, texdir)
    finally:
        base.VA, base.N = V0, N0


def _build(base, texdir):
    import wardrobe_more as M
    ctx = Ctx(base)
    if wardrobe_tex is not None:
        mats = wardrobe_tex.make_textures(texdir)
    else:
        mats = flat_materials(texdir)
    parts, hide = [], set()
    cm, info = cardigan(ctx)
    hide |= info["covered"]
    knit, rib, buttons = cardigan_parts(ctx, cm, info)
    parts.append(knit.part("cloth_cardigan", "cloth_knit", double_sided=True))
    parts.append(rib.part("cloth_cardigan_rib", "cloth_rib", double_sided=True))
    if buttons is not None:
        parts.append(button_mesh(ctx, *buttons, cm.P, cm.D).part("cloth_buttons", "cloth_buttons"))
    bl_shell, bl_collar, cov, _ = M.blouse(ctx, cm, info)
    hide |= cov
    parts.append(_merge(bl_shell, bl_collar).part("cloth_blouse", "cloth_blouse", double_sided=True))
    tr_shell, tr_bands, cov, _ = M.trousers(ctx)
    hide |= cov
    parts.append(_merge(tr_shell, tr_bands).part("cloth_trousers", "cloth_trousers", double_sided=True))
    sh_up, sh_sole, cov = M.shoes(ctx)
    hide |= cov
    parts.append(sh_up.part("shoes", "shoes", double_sided=True))
    parts.append(sh_sole.part("shoes_sole", "shoes", double_sided=True))
    set_weave_params(mats)
    return parts, mats, hide


def _merge(a, b):
    out = Acc()
    out.P, out.UV, out.D = a.P + b.P, a.UV + b.UV, a.D + b.D
    out.T = a.T + [t + a.n for t in b.T]
    out.weld = a.weld + [w + a.wn for w in b.weld]
    out.n, out.wn = a.n + b.n, a.wn + b.wn
    return out


WEAVE = {"cloth_knit": ("knit", 6.5), "cloth_rib": ("rib", 5.0), "cloth_blouse": ("plain", 1.0),
         "cloth_trousers": ("twill", 1.2)}
BAKED = os.environ.get("WARDROBE_BAKED", "0") == "1"


def set_weave_params(mats):
    """UVs are fabric metres / tile_m (own baked weave in albedo/normal), so the shader weave stays off."""
    for mid, (wv, st) in WEAVE.items():
        p = mats[mid]["params"]
        p.update({"weave": "none" if BAKED else wv, "stitch_mm": st, "weave_depth": 1.0, "uv_m_per_unit": p.get("tile_m", 1.0),
                  "weave_rot_deg": 0, "fuzz": {"cloth_knit": 0.7, "cloth_rib": 0.7}.get(mid, 0.25)})
