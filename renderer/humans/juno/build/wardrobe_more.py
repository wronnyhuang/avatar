"""asset — wardrobe part 2: blouse (+collar), trousers, shoes. Uses the helpers of wardrobe.py / wardrobe_geo.py."""
from __future__ import annotations

import numpy as np

from wardrobe_geo import QuadMesh, normalize, smoothstep, sweep, quad_normals, split_by_corner_uv


def cyl_corner_uv(B, Q, cx, cz, r_eff, v_of):
    """per-corner UV for a roughly vertical tube: u = angle * r_eff (metres), v = v_of(B) (metres); per-face unwrap."""
    X = B[Q]                                                    # (nf,4,3)
    ang = np.arctan2(X[..., 0] - cx, X[..., 2] - cz)
    ref = ang[:, :1]
    ang = np.where(ang - ref > np.pi, ang - 2 * np.pi, np.where(ref - ang > np.pi, ang + 2 * np.pi, ang))
    uv = np.zeros(Q.shape + (2,))
    uv[..., 0] = ang * r_eff
    uv[..., 1] = v_of(X.reshape(-1, 3)).reshape(Q.shape)
    return uv


def make_shell(ctx, fsel, off, subdiv=None):
    import wardrobe as W
    subdiv = W.SUBDIV if subdiv is None else subdiv
    """body faces -> QuadMesh (subdivided) with anchors/weights. returns (mesh, covered source face ids per child-mask fn)."""
    from wardrobe_geo import QuadMesh
    ids = ctx.body_fids[fsel]
    m = QuadMesh.from_body(ctx.base, ids, ctx.D)
    for _ in range(subdiv):
        m = m.subdivide()
    return m, ids


def settle(ctx, m, off, rounds=3, smooth_iters=3, pin=None):
    Nb = quad_normals(m.B, m.Q)
    P = m.B + Nb * np.broadcast_to(off, (len(m.B),))[:, None]
    for _ in range(rounds):
        P = m.smooth(P, iters=smooth_iters, pin=pin)
        P = ctx.sdf.push_out(P, np.asarray(off) * 0.85)
    return P


def covered_ids(ids, child_keep, subdiv=None):
    import wardrobe as W
    k = 4 ** (W.SUBDIV if subdiv is None else subdiv)
    return set(int(x) for x in ids[child_keep.reshape(-1, k).all(1)])


def _arm_w(ctx, Dc):
    return sum(Dc[:, ctx.bi[b]] for b in ctx.bi if b.split("_")[-1] in ("l", "r") and
               (b.startswith(("upperarm", "lowerarm", "hand", "thumb", "index", "middle", "ring", "pinky"))))


def _fold_noise(X, freq, seed):
    import wardrobe as W
    return W.noise3(X, freq, seed)


# ------------------------------------------------------------------------------------------------------------- trousers
def trousers(ctx):
    import wardrobe as W
    base = ctx.base
    V = base.VA
    Qb = ctx.body_Q
    cen = V[Qb].mean(1)
    Dc = ctx.D[Qb].mean(1)
    aw = _arm_w(ctx, Dc)
    headn = ctx.w(Dc, "head", "neck_01")
    y_w = ctx.y_thigh + 0.085
    foot_y = 0.5 * (ctx.bh["foot_l"][1] + ctx.bh["foot_r"][1])
    y_hem = foot_y + 0.045
    foot_w = ctx.w(Dc, "foot_l", "foot_r") + ctx.w(Dc, "toes_l", "toes_r") if "toes_l" in ctx.bi else ctx.w(Dc, "foot_l", "foot_r")
    sel = (cen[:, 1] > y_hem - 0.03) & (cen[:, 1] < y_w + 0.03) & (aw < 0.3) & (headn < 0.5) & (foot_w < 0.5)
    m, ids = make_shell(ctx, sel, 0.0)
    B = m.B
    keep = (B[:, 1] > y_hem) & (B[:, 1] < y_w)
    fk = keep[m.Q].sum(1) >= 3
    covered = covered_ids(ids, keep[m.Q].all(1))
    m = m.subset(fk)
    B = m.B
    knee_y = 0.5 * (ctx.bh["calf_l"][1] + ctx.bh["calf_r"][1])
    t_k = smoothstep(knee_y + 0.05, y_hem, B[:, 1])              # 0 above knee -> 1 at the ankle (straight leg flares off)
    off = 0.0075 + 0.016 * t_k
    P = settle(ctx, m, off, rounds=3, smooth_iters=4)
    # static folds: knee creases (front/back), ankle bunching
    N = quad_normals(P, m.Q)
    y = B[:, 1]
    kz = np.exp(-((y - knee_y) / 0.05) ** 2)
    n3 = _fold_noise(B, 5.0, 7)
    d = 0.0035 * kz * np.sin((y - knee_y) * 2 * np.pi / 0.035 + 1.5 * n3) * (np.abs(N[:, 2]) > 0.3)
    d += 0.004 * np.exp(-((y - (y_hem + 0.03)) / 0.04) ** 2) * np.sin(y * 2 * np.pi / 0.022 + 2 * _fold_noise(B, 6.0, 8))
    d += 0.0018 * _fold_noise(B, 9.0, 9) * smoothstep(y_hem, knee_y, y)
    P = ctx.sdf.push_out(P + N * d[:, None], 0.006)
    m.P = P
    m.D = m.lap_smooth(m.D, iters=10)
    # ---- UVs (per leg tube; crotch seam at x = 0) and the shell part
    left = B[:, 0] > 0
    hipc = 0.5 * (ctx.bh["thigh_l"] + ctx.bh["thigh_r"])
    uvs = np.zeros(m.Q.shape + (2,))
    fl = left[m.Q].sum(1) >= 2
    for side, fm in ((1, fl), (-1, ~fl)):
        cx = side * abs(ctx.bh["thigh_l"][0])
        cz = ctx.bh["thigh_l"][2]
        uvs[fm] = cyl_corner_uv(B, m.Q[fm], cx, cz, 0.07, lambda X: X[:, 1])
    tile = W.TILE("trousers")
    Pl, tris, UV, Dl, weld, _ = split_by_corner_uv(P, m.Q, uvs / tile, m.D)
    shell = W.Acc()
    shell.add(Pl, tris, UV, Dl, weld)
    # ---- hems + waistband as swept bands
    Ng = quad_normals(P, m.Q)
    st = W.surf_tree(P, Ng)
    bands = W.Acc()
    th = 0.004
    for lp in W.boundary_loops(m.Q):
        C = P[lp]
        top = C[:, 1].mean() > 0.5 * (y_hem + y_w)
        Cs = W.resample(W.smooth_chain(C, 15, closed=True), 0.008, closed=True)
        Cs, Ns = W.project_to(st, Cs)
        Ns = normalize(W.smooth_chain(Ns, 6, closed=True))
        Tn = normalize(np.roll(Cs, -1, 0) - np.roll(Cs, 1, 0))
        Bn = np.cross(Ns, Tn)
        sg = np.sign(np.mean(Bn[:, 1]))
        # hem: band hangs below the edge (turned hem 3 cm); waist: band hangs from the edge up
        if top:
            a0, a1 = (-0.012, 0.030) if sg > 0 else (-0.030, 0.012)
        else:
            a0, a1 = (-0.030, 0.022) if sg > 0 else (-0.022, 0.030)
        prof = W.band_profile(a0, a1, th, -0.002, r=0.0016, nr=1)
        Pp, tri, uv, Dd, wd = W.sweep_part_arrays(Cs, Ns, prof, True, tile, P, m.D)
        Pp = ctx.sdf.push_out(Pp, 0.006)
        bands.add(Pp, tri, uv, Dd, wd)
    return shell, bands, covered, dict(y_hem=y_hem, y_w=y_w)


# ---------------------------------------------------------------------------------------------------------------- blouse
COLLAR_T = 0.0022   # ASSET2 v6: collar leaf thickness (m)


def blouse(ctx, card_mesh, card_info):
    import wardrobe as W
    base = ctx.base
    V = base.VA
    Qb = ctx.body_Q
    cen = V[Qb].mean(1)
    Dc = ctx.D[Qb].mean(1)
    aw = _arm_w(ctx, Dc)
    y_w = ctx.y_thigh + 0.07
    neck_c = ctx.bh["neck_01"]
    sel = (cen[:, 1] > y_w - 0.03) & (aw < 0.5) & (ctx.w(Dc, "head") < 0.3) & (ctx.w(Dc, "thigh_l", "thigh_r") < 0.6) \
        & (cen[:, 1] < neck_c[1] + 0.06)
    m, ids = make_shell(ctx, sel, 0.0)
    B = m.B
    neck_w = ctx.w(m.D, "neck_01", "head")
    aw_v = _arm_w(ctx, m.D)
    keep = (B[:, 1] > y_w) & (neck_w < 0.3) & (aw_v < 0.5)
    fk = keep[m.Q].sum(1) >= 3
    covered = covered_ids(ids, keep[m.Q].all(1))
    m = m.subset(fk)
    B = m.B
    neck_w = ctx.w(m.D, "neck_01", "head")
    off = 0.0055 - 0.002 * smoothstep(0.0, 0.3, neck_w)
    P = settle(ctx, m, off, rounds=3, smooth_iters=3)
    N = quad_normals(P, m.Q)
    # soft vertical drape + waist gathers
    d = 0.0015 * _fold_noise(B, 8.0, 21) + 0.0012 * np.sin(B[:, 0] * 2 * np.pi / 0.06 + 2 * _fold_noise(B, 3.0, 22)) * (B[:, 1] < ctx.bust[1] - 0.03)
    P = ctx.sdf.push_out(P + N * d[:, None], 0.004)
    m.P = P
    m.D = m.lap_smooth(m.D, iters=10)
    tile = W.TILE("blouse")
    uvs = cyl_corner_uv(B, m.Q, 0.0, ctx.zc, 0.15, lambda X: X[:, 1])
    Pl, tris, UV, Dl, weld, _ = split_by_corner_uv(P, m.Q, uvs / tile, m.D)
    shell = W.Acc()
    shell.add(Pl, tris, UV, Dl, weld)
    # ---- collar: neck-edge loop (top boundary) -> two leaves lying over the cardigan neckline + a low stand at the back
    Ng = quad_normals(P, m.Q)
    st = W.surf_tree(P, Ng)
    loops = W.boundary_loops(m.Q)
    top = max(loops, key=lambda lp: P[lp, 1].mean())
    C = P[top]
    Cs = W.resample(W.smooth_chain(C, 60, closed=True), 0.0065, closed=True)
    cen_n = Cs.mean(0)
    ang = np.arctan2(Cs[:, 0] - cen_n[0], Cs[:, 2] - cen_n[2])    # 0 = front centre, +-pi = back centre
    n = len(Cs)
    cs = W.surf_tree(card_mesh.P, quad_normals(card_mesh.P, card_mesh.Q))
    uu, vv, UV_, Ds, T_, wd_ = [], [], [], [], [], []
    acc = W.Acc()
    for sgn in (1, -1):
        sel_i = np.where((ang * sgn >= 0.0) & (ang * sgn < np.pi))[0]
        if sgn < 0:
            sel_i = sel_i[::-1]
        # order from the back centre (ang ~ +-pi) to the front end (ang ~ 0)
        o = np.argsort(-np.abs(ang[sel_i]))
        ii = sel_i[o]
        ii = ii[np.abs(ang[ii]) > 0.30]                           # stop just short of the front centre
        top_pts = Cs[ii]
        s = (np.pi - np.abs(ang[ii])) / (np.pi - 0.30)            # 0 back -> 1 front tip
        # down direction: toward the chest along the shell + out
        nrm = normalize(np.stack([np.interp(top_pts[:, k], [0, 1], [0, 1]) * 0 for k in range(3)], 1) + 1e-9) if False else None
        _, Nt = W.project_to(st, top_pts)
        wdt = 0.016 + 0.075 * s ** 1.5                            # leaf length: stand at the back, long points at the front
        down = np.array([0, -1.0, 0])
        dir_ = normalize(down - Nt * (Nt @ down)[:, None])
        # shift toward the front centre as the leaf lengthens (points lean in)
        lat = np.array([-1.0, 0, 0]) * np.sign(top_pts[:, 0:1]) * (0.016 * s ** 2.0)[:, None]
        lat = lat - Nt * (lat * Nt).sum(1, keepdims=True)
        bot = top_pts + dir_ * wdt[:, None] + lat
        topl = top_pts
        rows = 5
        t = np.linspace(0, 1, rows)
        Pm = np.zeros((len(ii), rows, 3))
        for r in range(rows):
            Pm[:, r] = topl * (1 - t[r]) + bot * t[r]
        # lift: collar sits above the cardigan; fold roll near the neck edge
        roll = 0.006 * np.sin(np.pi * np.clip(t * 1.0, 0, 1))[None, :, None] ** 0.8 * 0.0 + 0.0
        flat = Pm.reshape(-1, 3)
        d_c, n_c, q_c = _sd_card(cs, flat)
        Nbody = W.project_to(st, flat)[1]
        want = 0.0045
        fl = flat + n_c * np.maximum(want - d_c, 0)[:, None]
        # a gentle rise at the neck edge (standing fold)
        rise = (0.5 + 0.5 * (1 - np.tile(t, len(ii))))[:, None] ** 3 * 0.005
        fl = fl + n_c * rise
        fl = ctx.sdf.push_out(fl, 0.016)
        Pm = fl.reshape(len(ii), rows, 3)
        for r in range(rows):
            Pm[:, r] = W.smooth_chain(Pm[:, r], 14)
        for c_ in range(len(ii)):
            Pm[c_] = W.smooth_chain(Pm[c_], 2)
        # double-sided thin leaf (two layers, top + under) for visible thickness
        cols = len(ii)
        vid = np.arange(cols * rows).reshape(cols, rows)
        Qs = np.array([(vid[i, j], vid[i + 1, j], vid[i + 1, j + 1], vid[i, j + 1]) for i in range(cols - 1) for j in range(rows - 1)])
        arc = np.r_[0, np.cumsum(np.linalg.norm(np.diff(top_pts, axis=0), axis=1))]
        UVl = np.stack([np.repeat(arc[:, None], rows, 1), np.repeat(t[None, :], cols, 0) * 0.04], -1).reshape(-1, 2) / tile
        tri = np.concatenate([Qs[:, [0, 1, 2]], Qs[:, [0, 2, 3]]])
        # orient outward (facing away from the body)
        fnn = np.cross(Pm.reshape(-1, 3)[tri[:, 1]] - Pm.reshape(-1, 3)[tri[:, 0]], Pm.reshape(-1, 3)[tri[:, 2]] - Pm.reshape(-1, 3)[tri[:, 0]])
        ref = Nbody[tri[:, 0]]
        if (np.sum(fnn * ref, 1) < 0).mean() > 0.5:
            tri = tri[:, ::-1]
        # ASSET2 v6 (judge r2: "flat white collar with no thickness"): a real under-layer COLLAR_T below the leaf + a rim
        # strip along the outer edge (bottom row + front point), so the leaf reads as folded cloth with a lit edge
        P1 = Pm.reshape(-1, 3)
        Nl = np.zeros_like(P1)
        fn = np.cross(P1[tri[:, 1]] - P1[tri[:, 0]], P1[tri[:, 2]] - P1[tri[:, 0]])
        for k in range(3):
            np.add.at(Nl, tri[:, k], fn)
        Nl = normalize(Nl + 1e-12)
        if (Nl * Nbody).sum(1).mean() < 0:
            Nl = -Nl
        P2 = P1 - Nl * COLLAR_T
        nv = len(P1)
        rim = [vid[i, rows - 1] for i in range(cols)] + [vid[cols - 1, j] for j in range(rows - 2, -1, -1)]
        rq = np.array([(rim[k], rim[k + 1], rim[k + 1] + nv, rim[k] + nv) for k in range(len(rim) - 1)])
        rtri = np.concatenate([rq[:, [0, 1, 2]], rq[:, [0, 2, 3]]])
        rc = 0.5 * (P1[rq[:, 0]] + P1[rq[:, 1]])
        PP = np.vstack([P1, P2])
        rf = np.cross(PP[rtri[:, 1]] - PP[rtri[:, 0]], PP[rtri[:, 2]] - PP[rtri[:, 0]])
        out_dir = np.concatenate([rc - P1.mean(0)] * 2)
        if (np.sum(rf * out_dir, 1) < 0).mean() > 0.5:
            rtri = rtri[:, ::-1]
        trall = np.concatenate([tri, tri[:, ::-1] + nv, rtri])
        UV2 = np.vstack([UVl, UVl])
        Dd = W.dense_from(P, m.D, PP)
        acc.add(PP, trall, UV2, Dd)
    # stand band over the jagged shell edge (low collar stand all round the neck)
    Cb = W.resample(W.smooth_chain(C, 40, closed=True), 0.0065, closed=True)
    Cb, Nb_ = W.project_to(st, Cb)
    Tn = normalize(np.roll(Cb, -1, 0) - np.roll(Cb, 1, 0))
    Bn = np.cross(Nb_, Tn)
    sgb = np.sign(np.mean(Bn[:, 1]))
    a0, a1 = (-0.022, 0.010) if sgb > 0 else (-0.010, 0.022)
    prof = W.band_profile(a0, a1, 0.0028, 0.0, r=0.0013, nr=1)
    Pp, tri, uv, Dd, wd = W.sweep_part_arrays(Cb, Nb_, prof, True, tile, P, m.D)
    Pp = ctx.sdf.push_out(Pp, 0.0045)
    acc.add(Pp, tri, uv, Dd, wd)
    return shell, acc, covered, dict(y_w=y_w)


def _sd_card(cs, X):
    tree, P, N = cs
    _, i = tree.query(X, k=1)
    return ((X - P[i]) * N[i]).sum(1), N[i], P[i]


# ----------------------------------------------------------------------------------------------------------------- shoes
def shoes_old(ctx):
    """flats: body-derived upper (foot faces below the ankle, offset), collar roll, welt + sole slab (extruded bottom)."""
    import wardrobe as W
    base = ctx.base
    V = base.VA
    Qb = ctx.body_Q
    cen = V[Qb].mean(1)
    Dc = ctx.D[Qb].mean(1)
    fw = sum(Dc[:, ctx.bi[b]] for b in ctx.bi if b.split("_")[-1] in ("l", "r") and b.startswith(("foot", "toe", "big", "ball", "heel")))
    fy = 0.5 * (ctx.bh["foot_l"][1] + ctx.bh["foot_r"][1])
    y_col = fy + 0.012                                        # collar height (flats: low, just under the ankle bone)
    sel = (fw > 0.5) & (cen[:, 1] < y_col + 0.03)
    m, ids = make_shell(ctx, sel, 0.0)
    B = m.B
    keep = B[:, 1] < y_col
    fk = keep[m.Q].sum(1) >= 3
    covered = covered_ids(ids, keep[m.Q].all(1))
    m = m.subset(fk)
    B = m.B
    Nb = quad_normals(B, m.Q)
    ymin = B[:, 1].min()
    sole_h = 0.011
    ball = ctx.w(m.D, "ball_l", "ball_r")
    off = 0.0075 + 0.006 * smoothstep(0.1, 0.5, ball)            # fuller toe box: bridges the toe grooves
    P = B + Nb * off[:, None]
    P = m.lap_smooth(P, iters=40)
    P = ctx.sdf.push_out(P, 0.0035)
    P = m.smooth(P, iters=8)
    P = ctx.sdf.push_out(P, 0.0035)
    m.P = P
    m.D = m.lap_smooth(m.D, iters=6)
    # ---- bottom faces -> sole slab (flattened copy dropped by sole_h), shell loses them
    N = quad_normals(P, m.Q)
    fN = N[m.Q].mean(1)
    fc = P[m.Q].mean(1)
    bot = (fN[:, 1] < -0.35)
    # flat sole plane under the lowest point; the sole copy's footprint follows the bottom faces
    y0 = P[:, 1].min() - 0.0005
    from scipy.sparse import coo_matrix  # noqa: F401
    Qbot = m.Q[bot]
    ub, inv = np.unique(Qbot, return_inverse=True)
    Qs = inv.reshape(-1, 4)
    Pb = P[ub].copy()
    # widen the footprint slightly (welt) radially from the local centroid line
    cxz = Pb[:, [0, 2]].mean(0)
    # per-vertex outward (use shell normal projected to xz for boundary vertices)
    Pt = Pb.copy()                                            # top of slab = original bottom faces (hidden under foot)
    Pd = Pb.copy()
    Pd[:, 1] = y0 - sole_h
    nb = N[ub]
    nx = np.stack([nb[:, 0], np.zeros(len(ub)), nb[:, 2]], 1)
    Pd += normalize(nx + 1e-9) * (0.0025 * np.linalg.norm(nx, axis=1, keepdims=True))
    Pt[:, 1] = np.minimum(Pt[:, 1], y0 + 0.002)
    # walls: boundary edges of the bottom patch
    E = np.concatenate([Qs[:, [0, 1]], Qs[:, [1, 2]], Qs[:, [2, 3]], Qs[:, [3, 0]]])
    key = np.sort(E, 1)
    u, i2, c = np.unique(key, axis=0, return_index=True, return_counts=True)
    bd = E[i2[c == 1]]
    nsl = len(ub)
    sole_P = np.vstack([Pd, P[ub]])                           # [0:n) bottom plate, [n:2n) rim top = original shell
    sole_T = []
    for q in Qs:
        sole_T += [(q[0], q[2], q[1]), (q[0], q[3], q[2])]       # bottom plate faces down
    for a, b in bd:
        sole_T += [(a, b, nsl + b), (a, nsl + b, nsl + a)]
    sole_T = np.array(sole_T)
    # orientation: make outward relative to the slab centroid
    cen_s = sole_P.mean(0)
    fnn = np.cross(sole_P[sole_T[:, 1]] - sole_P[sole_T[:, 0]], sole_P[sole_T[:, 2]] - sole_P[sole_T[:, 0]])
    cc = sole_P[sole_T].mean(1)
    sole_T = sole_T  # fixed below per-patch via double_sided anyway
    # shell without bottom faces
    sh_Q = m.Q[~bot]
    tile = W.TILE("leather")
    Ns = quad_normals(P, m.Q)
    # UV: planar-ish projection (top view for the upper), metres / tile
    uv_c = np.zeros(sh_Q.shape + (2,))
    X = P[sh_Q]
    uv_c[..., 0] = X[..., 0]
    uv_c[..., 1] = X[..., 2] + 0.35 * X[..., 1]
    Pl, tris, UV, Dl, weld, _ = split_by_corner_uv(P, sh_Q, uv_c / tile, m.D)
    upper = W.Acc()
    upper.add(Pl, tris, UV, Dl, weld)
    # collar roll around the opening
    st = W.surf_tree(P, Ns)
    sole = W.Acc()
    Dsl = W.dense_from(P, m.D, sole_P)
    sole.add(sole_P, sole_T, np.stack([sole_P[:, 0], sole_P[:, 2] + 0.2 * sole_P[:, 1]], 1) / tile, Dsl)
    for lp in W.boundary_loops(sh_Q):
        C = P[lp]
        if C[:, 1].mean() < y_col - 0.02:                      # the sole seam, not the opening
            continue
        Cs = W.resample(W.smooth_chain(C, 10, closed=True), 0.0065, closed=True)
        Cs, Nn = W.project_to(st, Cs)
        Tn = normalize(np.roll(Cs, -1, 0) - np.roll(Cs, 1, 0))
        Bn = np.cross(Nn, Tn)
        sg = np.sign(np.mean(Bn[:, 1]))
        a0, a1 = (-0.016, 0.004) if sg > 0 else (-0.004, 0.016)
        prof = W.band_profile(a0, a1, 0.0035, -0.001, r=0.0015, nr=1)
        Pp, tri, uv, Dd, wd = W.sweep_part_arrays(Cs, Nn, prof, True, tile, P, m.D)
        Pp = ctx.sdf.push_out(Pp, 0.0045)
        upper.add(Pp, tri, uv, Dd, wd)
    return upper, sole, covered


# ----------------------------------------------------------------------------------------------- shoes (hull sections)
def _hull_radius(poly, th):
    """poly (k,2) CCW convex hull about origin-centred coords; ray radius for angles th (angle from +y, clockwise toward +x)."""
    d = np.stack([np.sin(th), np.cos(th)], 1)
    A, Bp = poly, np.roll(poly, -1, 0)
    e = Bp - A
    best = np.full(len(th), np.inf)
    for k in range(len(A)):
        # solve t*d = A + u*e
        den = d[:, 0] * (-e[k, 1]) - d[:, 1] * (-e[k, 0])
        den = np.where(np.abs(den) < 1e-12, 1e-12, den)
        t = (A[k, 0] * (-e[k, 1]) - A[k, 1] * (-e[k, 0])) / den
        u = (d[:, 0] * A[k, 1] - d[:, 1] * A[k, 0]) / den
        ok = (t > 0) & (u >= -1e-9) & (u <= 1 + 1e-9)
        best = np.where(ok & (t < best), t, best)
    best[~np.isfinite(best)] = np.nanmin(best[np.isfinite(best)]) if np.isfinite(best).any() else 0.01
    return best


def _grid_tris(ns, nc, closed_cols=False):
    T = []
    for i in range(ns - 1):
        for j in range(nc - 1 + (1 if closed_cols else 0)):
            j1 = (j + 1) % nc
            a, b, c, d = i * nc + j, (i + 1) * nc + j, (i + 1) * nc + j1, i * nc + j1
            T += [(a, b, c), (a, c, d)]
    return np.array(T, np.int64)


def shoes(ctx):
    """flats built from convex-hull cross-sections of the foot (bridges the toe grooves): an upper (open collar +
    low throat, rim roll) and a sole slab with welt. Foot skin faces under the upper are hidden."""
    import wardrobe as W
    from scipy.spatial import ConvexHull
    base = ctx.base
    V = base.VA
    bodyv = base.verts("body")
    Dv = ctx.D[bodyv]
    Pv = V[bodyv]
    Qb = ctx.body_Q
    cen = V[Qb].mean(1)
    Dc = ctx.D[Qb].mean(1)
    isfoot = lambda D: sum(D[:, ctx.bi[b]] for b in ctx.bi if b.split("_")[-1] in ("l", "r") and b.startswith(("foot", "toe", "big", "ball", "heel")))
    fwv, fwc = isfoot(Dv), isfoot(Dc)
    y_col = 0.5 * (ctx.bh["foot_l"][1] + ctx.bh["foot_r"][1]) + 0.012
    up, sole = W.Acc(), W.Acc()
    covered = set()
    tile = W.TILE("leather")
    for side, sg in (("l", 1.0), ("r", -1.0)):
        sidem = (Pv[:, 0] * sg > 0)
        pts = Pv[(fwv > 0.5) & sidem & (Pv[:, 1] < y_col + 0.012)]
        a = ctx.bt["foot_" + side] - ctx.bh["foot_" + side]
        a = normalize(np.array([a[0], 0, a[2]]))
        lat = np.array([a[2], 0, -a[0]])
        org = pts.mean(0)
        s_all = (pts - org) @ a
        s0, s1 = s_all.min() - 0.004, s_all.max() + 0.009
        L = s1 - s0
        ymin = pts[:, 1].min()
        ns = 22   # ASSET3: lod1 budget (was 28)
        u = np.linspace(0, 1, ns)
        ss = s0 + (0.5 - 0.5 * np.cos(np.pi * u)) * L          # denser at the ends
        s_th = s0 + 0.54 * L                                     # throat: where the opening closes
        nc = 18   # ASSET3: lod1 budget (was 23)
        off = 0.0055
        R_all, C_all, ytop, ycut, thO = [], [], [], [], []
        for si in ss:
            q = pts[np.abs(s_all - si) < 0.009]
            if len(q) < 4:
                q = pts[np.argsort(np.abs(s_all - si))[:14]]
            l2 = np.stack([(q - org) @ lat, q[:, 1]], 1)
            h = ConvexHull(l2)
            poly = l2[h.vertices]
            c = 0.5 * (poly.min(0) + poly.max(0))
            C_all.append(c)
            R_all.append((poly - c))
            ytop.append(poly[:, 1].max())
        ytop = np.array(ytop)
        C_all = np.array(C_all)
        # end rounding: ring scale -> 0 over the last stretch
        Re_h, Re_t = 0.020, 0.024
        ec = np.sqrt(np.clip(1 - (np.clip(Re_h - (ss - s0), 0, None) / Re_h) ** 2, 0, 1)) * \
            np.sqrt(np.clip(1 - (np.clip(Re_t - (s1 - ss), 0, None) / Re_t) ** 2, 0, 1))
        ec = np.maximum(ec, 0.0)
        th_f = np.linspace(0, 2 * np.pi, 361)
        rows, ycuts = [], []
        for i, si in enumerate(ss):
            r = _hull_radius(R_all[i], th_f) + off
            ring = np.stack([np.sin(th_f) * r, np.cos(th_f) * r], 1) + C_all[i]
            ring[:, 1] = np.maximum(ring[:, 1], ymin + 0.002)
            ss_ = smoothstep(s_th - 0.07, s_th + 0.03, si)
            ycut = min(y_col - 0.014, ytop[i] + off - 0.022)
            # opening half-angle: first theta (from the top, going clockwise) where the ring drops below ycut
            top = ring[:181]; topL = ring[::-1][:181]
            below = np.where(top[:, 1] <= ycut)[0]
            below2 = np.where(ring[::-1][:181, 1] <= ycut)[0]
            k1 = below[0] if len(below) else 0
            k2 = below2[0] if len(below2) else 0
            ko = int(max(k1, k2) * (1 - ss_) ** 0.8)
            ths = np.linspace(th_f[ko], 2 * np.pi - th_f[ko], nc)
            rr = _hull_radius(R_all[i], ths) + off
            pr = np.stack([np.sin(ths) * rr, np.cos(ths) * rr], 1) * ec[i] + C_all[i]
            pr[:, 1] = np.maximum(pr[:, 1], ymin + 0.002)
            rows.append(pr)
            ycuts.append(ycut)
        rows = np.array(rows)                                     # (ns, nc, 2)  (lateral, y)
        sA = ss[:, None]
        P = org[None, None, :] * np.array([1, 0, 1]) + a[None, None, :] * sA[..., None] + lat[None, None, :] * rows[..., 0:1] + \
            np.array([0, 1, 0])[None, None, :] * rows[..., 1:2]
        Pf = P.reshape(-1, 3)
        T = _grid_tris(ns, nc)
        # outward orientation test
        cc = Pf[T].mean(1)
        fn = np.cross(Pf[T[:, 1]] - Pf[T[:, 0]], Pf[T[:, 2]] - Pf[T[:, 0]])
        ctr = np.repeat(P.mean(1), nc, axis=0)[T[:, 0]]
        if (((cc - ctr) * fn).sum(1) < 0).mean() > 0.5:
            T = T[:, ::-1]
        # smooth lightly and keep clearance
        Pf = ctx.sdf.push_out(Pf, 0.0035)
        UV = np.stack([(Pf @ lat), (Pf @ a) + 0.3 * Pf[:, 1]], 1) / tile
        Dd = W.dense_from(Pv, Dv, Pf)
        up.add(Pf, T, UV, Dd)
        # ---- collar rim roll along the opening edge (col 0 down the left side, col nc-1 back)
        E1 = P[:, 0, :]
        E2 = P[:, nc - 1, :]
        edge = np.vstack([E1[::1], E2[::-1]])
        keep = np.r_[True, np.linalg.norm(np.diff(edge, axis=0), axis=1) > 1e-5]
        # drop the closed (forward) stretch where the opening has zero width
        wopen = np.linalg.norm(E1 - E2, axis=1)
        sel_rows = np.where(wopen > 0.004)[0]
        if len(sel_rows) > 3:
            r0, r1 = sel_rows[0], sel_rows[-1]
            e1, e2 = E1[r0:r1 + 1], E2[r0:r1 + 1]
            # include the closed ends so the loop is a proper loop
            loop = np.vstack([e1, e2[::-1]])
            Cs = W.resample(W.smooth_chain(loop, 6, closed=False), 0.005, closed=False)
            prof = W.band_profile(-0.0035, 0.0035, 0.0035, -0.0017, r=0.0017, nr=1)
            nrm_ = ctx.sdf.query(Cs)[1]
            Pp, tri, uv, Dd2, wd = W.sweep_part_arrays(Cs, nrm_, prof, False, tile, Pv, Dv)
            Pp = ctx.sdf.push_out(Pp, 0.0035)
            up.add(Pp, tri, uv, Dd2, wd)
        # ---- hidden skin: foot faces under the upper (below the opening cut)
        fm = (fwc > 0.5) & (cen[:, 0] * sg > 0) & (cen[:, 1] < y_col + 0.03)
        ids = ctx.body_fids[fm]
        sc = (cen[fm] - org) @ a
        ycf = np.interp(sc, ss, np.minimum(np.array(ycuts), 1e3 * 0 + 10))
        fv = Qb[fm]
        vmaxy = V[fv][..., 1].max(1)
        covered |= set(int(x) for x in ids[vmaxy < np.minimum(ycf, y_col) - 0.002])
        # ---- sole slab: swept superellipse per station
        y_top = ymin + 0.005
        y_bot = ymin - 0.001
        nsl = 14
        ang = np.linspace(0, 2 * np.pi, nsl, endpoint=False)
        from scipy.ndimage import gaussian_filter1d
        low = rows[:, :, 1] < ymin + 0.016
        wl = rows[:, :, 0]
        wmax = np.array([wl[i][low[i]].max() if low[i].any() else wl[i].max() for i in range(ns)])
        wmin = np.array([wl[i][low[i]].min() if low[i].any() else wl[i].min() for i in range(ns)])
        wmax, wmin = gaussian_filter1d(wmax, 1.5, mode="nearest"), gaussian_filter1d(wmin, 1.5, mode="nearest")
        half = 0.5 * (wmax - wmin) + 0.0025
        lc = 0.5 * (wmax + wmin)
        ec2 = ec * 0 + np.sqrt(np.clip(1 - (np.clip(Re_h - (ss - s0), 0, None) / Re_h) ** 2, 0, 1)) * \
            np.sqrt(np.clip(1 - (np.clip(Re_t - (s1 - ss), 0, None) / Re_t) ** 2, 0, 1))
        half = np.maximum(half * ec2, 1e-5)
        cx = np.sign(np.cos(ang)) * np.abs(np.cos(ang)) ** 0.3
        sy = np.sign(np.sin(ang)) * np.abs(np.sin(ang)) ** 0.55
        S = np.zeros((ns, nsl, 3))
        for i in range(ns):
            thick_f = 0.15 + 0.85 * ec2[i]
            ym = 0.5 * (y_top + y_bot)
            S[i] = a * ss[i] + org * np.array([1, 0, 1]) + lat[None, :] * (lc[i] + half[i] * cx)[:, None] + \
                np.array([0, 1, 0])[None, :] * (ym + 0.5 * (y_top - y_bot) * sy * thick_f + (0 if True else 0))[:, None]
        Sf = S.reshape(-1, 3)
        Ts = _grid_tris(ns, nsl, closed_cols=True)
        cc = Sf[Ts].mean(1)
        fn = np.cross(Sf[Ts[:, 1]] - Sf[Ts[:, 0]], Sf[Ts[:, 2]] - Sf[Ts[:, 0]])
        ctr = np.repeat(S.mean(1), nsl, axis=0)[Ts[:, 0]]
        if (((cc - ctr) * fn).sum(1) < 0).mean() > 0.5:
            Ts = Ts[:, ::-1]
        UVs = np.stack([(Sf @ lat), (Sf @ a) + 0.2 * Sf[:, 1]], 1) / tile
        sole.add(Sf, Ts, UVs, W.dense_from(Pv, Dv, Sf))
    return up, sole, covered
