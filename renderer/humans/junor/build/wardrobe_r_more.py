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
    import wardrobe_r as W
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
    import wardrobe_r as W
    k = 4 ** (W.SUBDIV if subdiv is None else subdiv)
    return set(int(x) for x in ids[child_keep.reshape(-1, k).all(1)])


def _arm_w(ctx, Dc):
    return sum(Dc[:, ctx.bi[b]] for b in ctx.bi if b.split("_")[-1] in ("l", "r") and
               (b.startswith(("upperarm", "lowerarm", "hand", "thumb", "index", "middle", "ring", "pinky"))))


def _fold_noise(X, freq, seed):
    import wardrobe_r as W
    return W.noise3(X, freq, seed)


# ------------------------------------------------------------------------------------------------------------- trousers
def trousers(ctx):
    import wardrobe_r as W
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
COLLAR_T = 0.0024      # collar ply thickness (m)
STAND_CLR = 0.0045     # stand off the neck skin
STAND_BELOW = 0.007    # stand reaches this far below the collar-base saddle (covers the shell edge)
FOLD_R = 0.0032        # fold radius at the top of the stand
GAP_TH = 0.26          # half angle of the open front (rad from the front centre); round 3: open collar
COLLAR_POINT = 0.034   # extra fall depth at the points
COLLAR_SPREAD = 0.32   # sideways lean of the points per metre of fall (open collar: points spread onto the fronts)
COLLAR_CLOSED = False  # round 2 closed the stand across the front + top button; the lead's round 3 wants it open
HS_FRONT, HS_BACK = 0.0125, 0.025   # stand height at the front ends / centre back (round 3: lower at the front/sides)
FALL_EXTRA = 0.013     # fall depth beyond the stand at the back
PLACKET_V = 0.050      # open placket: V bottom this far below the collar base front point
PLACKET_XPAD = 0.012   # V half width at the top beyond the collar ends
PLACKET_WELD_CLR = -1.0  # (off) set >0 to keep the V bottom above the weld loop if its seam ever shows
PLACKET_W = 0.011      # width of the turned-back front edge along the V
COLLAR_COLS = 116      # columns along the collar (lead 11-16: outline wavy at 1080p with 76)
EDGE_SMOOTH = 24       # smoothing of the outer outline along the collar after the lay-over
FALL_CLR = 0.0046      # fall clearance over every layer it lies on (> COLLAR_T)
FALL_DEPTH = 0.009     # only lift fall points that are at most this far under a layer
FALL_STIFF = 6        # displacement smoothing per lay-over round (stiffer cloth bridges the rib-band step)


def blouse(ctx, card_mesh, card_info, card_parts=()):
    import wardrobe_r as W
    import wardrobe_r_neck as NK
    base = ctx.base
    nf = ctx.nf
    V = base.VA
    Qb = ctx.body_Q
    cen = V[Qb].mean(1)
    Dc = ctx.D[Qb].mean(1)
    aw = _arm_w(ctx, Dc)
    y_w = ctx.y_thigh + 0.07
    sel = (cen[:, 1] > y_w - 0.03) & (aw < 0.5) & (ctx.w(Dc, "head") < 0.3) & (ctx.w(Dc, "thigh_l", "thigh_r") < 0.6) \
        & (cen[:, 1] < nf.y_side + 0.04)
    m, ids = make_shell(ctx, sel, 0.0)
    B = m.B
    aw_v = _arm_w(ctx, m.D)
    keep = (B[:, 1] > y_w) & ~nf.above(B, -0.003) & (aw_v < 0.5)
    fk = keep[m.Q].sum(1) >= 3
    # open neck (MH quads are ~2 cm, coarser than the V): drop faces whose centre is inside the V, then snap the new hole's
    # boundary vertices onto the V lines (the placket band covers what is left of the stair-steps)
    fin = _placket_cut(ctx, m.B[m.Q].mean(1))
    fk &= ~fin
    covered = covered_ids(ids, keep[m.Q].all(1) & ~fin)
    m = m.subset(fk)
    if not COLLAR_CLOSED:
        xg, yf, yv = _placket_geom(ctx)
        bd = m.boundary_mask()
        Bm = m.B
        near = bd & (Bm[:, 2] > ctx.zc + 0.02) & (Bm[:, 1] > yv - 0.015) & (Bm[:, 1] < yf - 0.004) & (np.abs(Bm[:, 0]) < xg + 0.025)
        if near.any():
            X = Bm[near]
            q = _closest_on_v(X[:, :2], xg, yf, yv)
            Xn = np.c_[q, X[:, 2]]
            sd, nn_, _ = ctx.sdf.query(Xn)
            m.B[near] = Xn - nn_ * sd[:, None]
    B = m.B
    nn = ctx.near_neck(B, 0.05)
    off = 0.0050 - 0.0020 * nn
    P = settle(ctx, m, off, rounds=3, smooth_iters=3)
    N = quad_normals(P, m.Q)
    d = 0.0012 * _fold_noise(B, 8.0, 21) * (1 - nn) + 0.0010 * np.sin(B[:, 0] * 2 * np.pi / 0.06 + 2 * _fold_noise(B, 3.0, 22)) * (B[:, 1] < ctx.bust[1] - 0.03)
    P = ctx.sdf.push_out(P + N * d[:, None], 0.003)
    P = ctx.skv.push_out(P, 0.0025)
    m.P = P
    m.D = m.lap_smooth(m.D, iters=10)
    tile = W.TILE("blouse")
    uvs = cyl_corner_uv(B, m.Q, 0.0, ctx.zc, 0.15, lambda X: X[:, 1])
    Pl, tris, UV, Dl, weld, _ = split_by_corner_uv(P, m.Q, uvs / tile, m.D)
    shell = W.Acc()
    shell.add(Pl, tris, UV, Dl, weld)
    pl = None if COLLAR_CLOSED else _placket(ctx, m, tile)
    acc = collar(ctx, m, card_parts, tile, extra_layers=[] if pl is None else [(np.concatenate(pl.P), np.concatenate(pl.T))])
    if pl is not None:
        acc = W._merge(acc, pl)
    return shell, acc, covered, dict(y_w=y_w)


DEBUG = {}


def _placket_geom(ctx):
    """(x half width at the collar base, y of the collar base front, y of the V bottom, z ref)."""
    nf = ctx.nf
    P, _ = nf.curve(np.array([GAP_TH, -GAP_TH]))
    xg = float(np.abs(P[:, 0]).mean()) + PLACKET_XPAD
    yf = float(P[:, 1].mean())
    # the V bottom stays above the MH/ICT weld (its seam must not show inside the open neck)
    H = ctx.base.ict
    L = H.P[H.neck_loop]
    y_weld = float(L[np.argmin(np.abs(L[:, 0]) + 10 * (L[:, 2] < 0)), 1])
    return xg, yf, max(yf - PLACKET_V, y_weld + PLACKET_WELD_CLR)


def _closest_on_v(XY, xg, yf, yv):
    """closest points on the two V segments (0, yv) -> (+-xg, yf), choosing the segment on the point's own side."""
    out = np.zeros_like(XY)
    for k, (x, y) in enumerate(XY):
        a = np.array([0.0, yv])
        b = np.array([np.sign(x) * xg if x != 0 else xg, yf])
        d = b - a
        t = np.clip(((np.array([x, y]) - a) @ d) / (d @ d), 0, 1)
        out[k] = a + t * d
    return out


def _placket_cut(ctx, B):
    """open neck: blouse shell vertices inside the V (front, between the V lines from the collar ends to the V bottom)."""
    if COLLAR_CLOSED:
        return np.zeros(len(B), bool)
    xg, yf, yv = _placket_geom(ctx)
    t = np.clip((B[:, 1] - yv) / (yf - yv), 0, None)
    return (B[:, 2] > ctx.zc + 0.02) & (B[:, 1] > yv) & (np.abs(B[:, 0]) < xg * t) & (B[:, 1] < yf + 0.04)


def _placket(ctx, m, tile):
    """the turned-back front edges of the open neck: a thin fabric band along the V (covers the shell's cut, gives the V
    a clean folded edge; skin shows inside the V)."""
    import wardrobe_r as W
    xg, yf, yv = _placket_geom(ctx)
    n = 26
    s = np.linspace(-1, 1, 2 * n + 1)
    x = xg * s
    y = yv + (yf - yv) * np.abs(s)
    y = y + 0.004                                       # top ends tuck under the collar stand ends
    Ng = quad_normals(m.P, m.Q)
    cand = np.where(m.P[:, 2] > ctx.zc)[0]
    from scipy.spatial import cKDTree
    kd = cKDTree(m.P[cand][:, :2])
    _, j = kd.query(np.c_[x, y], k=4)
    z = m.P[cand][j][:, :, 2].mean(1)
    C = np.c_[x, y, z]
    st = W.surf_tree(m.P, Ng)
    C = W.smooth_chain(C, 3)
    C, Nc = W.project_to(st, C)
    C = W.resample(C, 0.004)
    C, Nc = W.project_to(st, C)
    Tn = normalize(np.gradient(C, axis=0))
    Bn = np.cross(Nc, Tn)
    sg = -np.sign(np.mean(Bn[:, 0] * -np.sign(C[:, 0] + 1e-9)))   # + = Bn points away from the open V (shell side)
    a0, a1 = (-0.0015, PLACKET_W) if sg > 0 else (-PLACKET_W, 0.0015)
    prof = W.band_profile(a0, a1, 0.0022, -0.0004, r=0.0009, nr=2)
    P, tri, uv, D, wd = W.sweep_part_arrays(C, Nc, prof, False, tile, m.P, m.D)
    P = ctx.skv.push_out(P, 0.0015)
    acc = W.Acc()
    acc.add(P, tri, uv, D, wd)
    return acc


def _grid_quads(C, R):
    vid = np.arange(C * R).reshape(C, R)
    return vid, np.array([(vid[i, j], vid[i + 1, j], vid[i + 1, j + 1], vid[i, j + 1])
                          for i in range(C - 1) for j in range(R - 1)], np.int64)


def _grid_lap(G, pin_rows, iters, lam=0.45, row_only=False):
    """laplacian smoothing on a (C, R, 3) grid; rows in pin_rows fixed, the outer border rows/cols use 1D neighbours."""
    G = G.copy()
    C, R = G.shape[:2]
    free = np.ones(R, bool)
    free[list(pin_rows)] = False
    for _ in range(iters):
        L = np.zeros_like(G)
        L[1:-1] += 0.5 * (G[:-2] + G[2:]) - G[1:-1]            # along the collar
        L[0] += G[1] - G[0]                                     # free ends (Neumann)
        L[-1] += G[-2] - G[-1]
        if not row_only:
            L[:, 1:-1] += 0.5 * (G[:, :-2] + G[:, 2:]) - G[:, 1:-1]
            L[:, -1] += G[:, -2] - G[:, -1]
        L[:, ~free] = 0
        G += lam * L
    return G


def collar(ctx, shell_m, card_parts, tile, extra_layers=()):
    """shirt collar on the true neck: a stand rising from the collar-base saddle, a rounded fold, and a fall that turns down
    over the stand and lies on the cardigan neckline / blouse front, ending in two points at the open front."""
    import wardrobe_r as W
    import wardrobe_r_neck as NK
    nf, sk = ctx.nf, ctx.sk
    ncol = COLLAR_COLS
    th = np.linspace(GAP_TH, 2 * np.pi - GAP_TH, ncol)
    th = np.where(th > np.pi, th - 2 * np.pi, th)
    Pk, Nk = nf.curve(th)
    Pu, _ = nf.curve(th, drop=-0.02)
    Pk = W.smooth_chain(Pk, 4)
    Pu = W.smooth_chain(Pu, 4)
    U = normalize(Pu - Pk)
    T = normalize(np.gradient(Pk, axis=0))
    n_st = normalize(np.cross(U, T))
    rad = np.stack([np.sin(th), np.zeros_like(th), np.cos(th)], 1)
    n_st *= np.where((n_st * rad).sum(1) < 0, -1.0, 1.0)[:, None]
    U = normalize(np.cross(T, n_st)) * np.sign(np.mean(np.cross(T, n_st)[:, 1]))
    ath = np.abs(th)
    front = smoothstep(1.35, GAP_TH + 0.05, ath)                 # 1 at the collar points, 0 from the sides back
    hs = HS_FRONT + (HS_BACK - HS_FRONT) * (ath / np.pi) ** 2.0   # stand: 2.5 cm at the back, ~1.6 at the sides, 1.25 front
    A = Pk + n_st * STAND_CLR - U * STAND_BELOW
    # ---- rows: stand (4), fold (3), fall (9)
    rows = []
    for t in np.linspace(0, 1, 4):
        rows.append(A + U * ((hs + STAND_BELOW) * t)[:, None])
    top = rows[-1]
    for a in (np.pi / 3, 2 * np.pi / 3, np.pi):
        rows.append(top + n_st * FOLD_R * (1 - np.cos(a)) + U * FOLD_R * np.sin(a))
    F0 = rows[-1]
    w = hs + STAND_BELOW + FALL_EXTRA + COLLAR_POINT * front ** 1.6            # fall depth: covers the stand + ~1 cm; points ~6 cm
    nfall = 8
    sx = np.sign(Pk[:, 0:1])
    for k in range(1, nfall + 1):
        s = w * k / nfall
        beyond = np.maximum(s - (hs + STAND_BELOW), 0)
        X = F0 - U * s[:, None] + n_st * (0.0015 * k / nfall)
        X = X + np.array([1.0, 0, 0]) * sx * (COLLAR_SPREAD * beyond * front)[:, None]    # the points spread outward
        rows.append(X)
    G = np.stack(rows, 1)                                          # (C, R, 3)
    C, R = G.shape[:2]
    ns = 4
    fall0 = ns + 3
    # stand: clear the true skin
    G[:, :ns] = sk.push_out(G[:, :ns].reshape(-1, 3), STAND_CLR * 0.8).reshape(C, ns, 3)
    # layers under the fall: skin, blouse shell, cardigan shell + rib, the stand's outer face
    Ly = NK.Layers(sk)
    Ly.add(ctx.skv.P, ctx.skv.T)
    Ly.add(shell_m.P, np.concatenate([shell_m.Q[:, [0, 1, 2]], shell_m.Q[:, [0, 2, 3]]]))
    for p in card_parts:
        Ly.add(p.P, p.tris)
    for P_, T_ in extra_layers:
        Ly.add(P_, T_)
    vid_s, Qs = _grid_quads(C, ns)
    st_out = (G[:, :ns] + n_st[:, None, :] * COLLAR_T).reshape(-1, 3)
    Ly.add(st_out, np.concatenate([Qs[:, [0, 1, 2]], Qs[:, [0, 2, 3]]]), lat=0.006)
    # lay the fall over the layers: push, then smooth the DISPLACEMENT field (positions are never smoothed, so the
    # free outer edge does not shrink toward the fold)
    X0 = G[:, fall0:].copy()
    X = X0.copy()
    for it in range(10):
        Xp = Ly.push(X.reshape(-1, 3), FALL_CLR, depth=FALL_DEPTH).reshape(X.shape)
        D = Xp - X0
        D = _grid_lap(np.concatenate([np.zeros_like(D[:, :1]), D], 1), [0], iters=FALL_STIFF)[:, 1:]
        X = X0 + D
    # the outline itself: smooth the outer rows ALONG the collar (ends pinned, so the points stay sharp), more at the edge
    nr = X.shape[1]
    for j in range(max(0, nr - 5), nr):
        wj = (j - (nr - 6)) / 5.0
        Xs = X[:, j].copy()
        for _ in range(int(EDGE_SMOOTH * wj)):
            Xs[1:-1] += 0.5 * (0.5 * (Xs[:-2] + Xs[2:]) - Xs[1:-1])
        X[:, j] = Xs
    # final lay-over rounds with a SMOOTHED lift (a raw per-vertex push left a torn-looking edge at 1080p)
    for _ in range(4):
        Xp = Ly.push(X.reshape(-1, 3), FALL_CLR, depth=FALL_DEPTH).reshape(X.shape)
        D = Xp - X
        D = _grid_lap(np.concatenate([D[:, :1] * 0, D], 1), [0], iters=4)[:, 1:]
        D = D * 1.25                                         # slight over-lift: the smoothing averages pushes down
        X = X + D
    G[:, fall0:] = X
    DEBUG["G"], DEBUG["ns"], DEBUG["fall0"], DEBUG["n_st"] = G.copy(), ns, fall0, n_st
    # ---- mesh: outer ply + inner ply (offset along the sheet normal, outward on the stand = inward on the fall) + rim
    vid, Q = _grid_quads(C, R)
    P1 = G.reshape(-1, 3)
    tri = np.concatenate([Q[:, [0, 1, 2]], Q[:, [0, 2, 3]]])
    Nl = W.tri_normals(P1, tri)
    if (Nl[vid[:, 1]] * n_st).sum(1).mean() < 0:
        tri = tri[:, ::-1]
        Nl = -Nl
    P2 = P1 + Nl * COLLAR_T
    nv = len(P1)
    ring = list(vid[:, 0]) + list(vid[-1, 1:]) + list(vid[::-1, -1][1:]) + list(vid[0, ::-1][1:-1])
    ring = np.array(ring)
    rq = np.stack([ring, np.roll(ring, -1), np.roll(ring, -1) + nv, ring + nv], 1)
    rtri = np.concatenate([rq[:, [0, 1, 2]], rq[:, [0, 2, 3]]])
    PP = np.vstack([P1, P2])
    trall = np.concatenate([tri, tri[:, ::-1] + nv, rtri])
    arc = np.r_[0, np.cumsum(np.linalg.norm(np.diff(Pk, axis=0), axis=1))]
    acr = np.concatenate([np.zeros((C, 1)), np.cumsum(np.linalg.norm(np.diff(G, axis=1), axis=2), 1)], 1)
    UVl = np.stack([np.repeat(arc[:, None], R, 1), acr], -1).reshape(-1, 2) / tile
    UV2 = np.vstack([UVl, UVl])
    # weights: the stand follows the neck skin, the fall the shoulders below it
    # weights: the stand/fold follow the ICT neck skin's own weights (what the collar wraps), the fall the visible skin under it
    Psrc, Dsrc = _weight_source(ctx)
    Dd = W.dense_from(Psrc, Dsrc, _anchor(G, Pk, ns, fall0))
    Dd = np.vstack([Dd, Dd])
    acc = W.Acc()
    acc.add(PP, trall, UV2, Dd)
    # ---- the closed front of the stand (lead 11-16: the open gap read as a dark triangle): the band continues across
    # the throat between the two points, buttoned, so the V between the points shows white stand, not shadowed skin
    DEBUG.pop("stand_button", None)
    if not COLLAR_CLOSED:
        return acc
    acc_f = _front_stand(ctx, th, Pk, hs, tile, Psrc, Dsrc)
    out = W.Acc()
    for a_ in (acc, acc_f):
        out.add(np.concatenate(a_.P), np.concatenate(a_.T), np.concatenate(a_.UV), np.concatenate(a_.D))
    return out


def _stand_frame(nf, th):
    import wardrobe_r as W
    Pk, _ = nf.curve(th)
    Pu, _ = nf.curve(th, drop=-0.02)
    U = normalize(Pu - Pk)
    T = normalize(np.gradient(Pk, axis=0))
    n_st = normalize(np.cross(U, T))
    rad = np.stack([np.sin(th), np.zeros_like(th), np.cos(th)], 1)
    n_st *= np.where((n_st * rad).sum(1) < 0, -1.0, 1.0)[:, None]
    U = normalize(np.cross(T, n_st)) * np.sign(np.mean(np.cross(T, n_st)[:, 1]))
    return Pk, U, n_st


def _front_stand(ctx, th_main, Pk_main, hs_main, tile, Psrc, Dsrc):
    """stand band across the open front: |theta| <= GAP_TH, same construction as the collar's stand rows, its ends meeting
    the collar's first/last columns; rounded top (the fold), two plies + rim, a buttonhole-side overlap line."""
    import wardrobe_r as W
    nf, sk = ctx.nf, ctx.sk
    nc = 15
    th = np.linspace(-GAP_TH, GAP_TH, nc)
    # use the collar's own end frames at +-GAP so the seams line up
    Pk, U, n_st = _stand_frame(nf, np.r_[th_main[-3:], th, th_main[:3]])
    Pk = W.smooth_chain(Pk, 4)[3:-3]
    U, n_st = U[3:-3], n_st[3:-3]
    # blend the curve ends onto the collar's (smoothed) end points
    Pk[0] = Pk_main[-1]
    Pk[-1] = Pk_main[0]
    hs = np.full(nc, float(min(hs_main[0], hs_main[-1])))
    A = Pk + n_st * STAND_CLR - U * STAND_BELOW
    rows = [A + U * ((hs + STAND_BELOW) * t)[:, None] for t in np.linspace(0, 1, 4)]
    top = rows[-1]
    for a in (np.pi / 3, 2 * np.pi / 3):
        rows.append(top + n_st * FOLD_R * (1 - np.cos(a)) + U * FOLD_R * np.sin(a))
    G = np.stack(rows, 1)
    C, R = G.shape[:2]
    G[:, :4] = sk.push_out(G[:, :4].reshape(-1, 3), STAND_CLR * 0.8).reshape(C, 4, 3)
    vid, Q = _grid_quads(C, R)
    P1 = G.reshape(-1, 3)
    tri = np.concatenate([Q[:, [0, 1, 2]], Q[:, [0, 2, 3]]])
    Nl = W.tri_normals(P1, tri)
    if (Nl[vid[:, 1]] * n_st).sum(1).mean() < 0:
        tri = tri[:, ::-1]
        Nl = -Nl
    P2 = P1 - Nl * COLLAR_T          # inner ply toward the neck (the outer face is the visible one)
    nv = len(P1)
    ring = np.array(list(vid[:, 0]) + list(vid[-1, 1:]) + list(vid[::-1, -1][1:]) + list(vid[0, ::-1][1:-1]))
    rq = np.stack([ring, np.roll(ring, -1), np.roll(ring, -1) + nv, ring + nv], 1)
    rtri = np.concatenate([rq[:, [0, 1, 2]], rq[:, [0, 2, 3]]])
    PP = np.vstack([P1, P2])
    trall = np.concatenate([tri, tri[:, ::-1] + nv, rtri])
    arc = np.r_[0, np.cumsum(np.linalg.norm(np.diff(Pk, axis=0), axis=1))]
    acr = np.concatenate([np.zeros((C, 1)), np.cumsum(np.linalg.norm(np.diff(G, axis=1), axis=2), 1)], 1)
    UVl = np.stack([np.repeat(arc[:, None], R, 1), acr], -1).reshape(-1, 2) / tile
    D = W.dense_from(Psrc, Dsrc, np.repeat(Pk, R, 0))
    acc = W.Acc()
    acc.add(PP, trall, np.vstack([UVl, UVl]), np.vstack([D, D]))
    # top button of the stand (blouse fabric material keeps the part list unchanged): a small flat disc, proud of the band
    c = A[nc // 2] + U[nc // 2] * (0.5 * (hs[0] + STAND_BELOW)) + n_st[nc // 2] * (COLLAR_T + 0.0006)
    DEBUG["stand_button"] = (c, n_st[nc // 2])
    return acc


def _weight_source(ctx):
    """points + dense weights of the skin that is rendered: MH body minus cut faces, and the ICT face/neck skin."""
    base = ctx.base
    H = base.ict
    keep = sorted({v for fi in base.faces_of("body") if fi not in base.cut_faces for v, _ in base.obj.faces[fi]})
    sk_ids = np.concatenate([H.region("face"), H.region("headneck")])
    J, Wt = base.ict_JW
    Di = np.zeros((len(sk_ids), ctx.D.shape[1]))
    for k in range(J.shape[1]):
        np.add.at(Di, (np.arange(len(sk_ids)), J[sk_ids, k].astype(np.int64)), Wt[sk_ids, k])
    return np.vstack([ctx.base.VA[keep], H.P[sk_ids]]), np.vstack([ctx.D[keep], Di])


def _anchor(G, Pk, ns, fall0):
    """skin anchor per collar grid vertex: the stand/fold at the neckline point, the fall at its own position."""
    C, R = G.shape[:2]
    A = np.repeat(Pk[:, None, :], R, 1).copy()
    A[:, fall0:] = G[:, fall0:]
    return A.reshape(-1, 3)


def _sd_card(cs, X):
    tree, P, N = cs
    _, i = tree.query(X, k=1)
    return ((X - P[i]) * N[i]).sum(1), N[i], P[i]


# ----------------------------------------------------------------------------------------------------------------- shoes
def shoes_old(ctx):
    """flats: body-derived upper (foot faces below the ankle, offset), collar roll, welt + sole slab (extruded bottom)."""
    import wardrobe_r as W
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
    import wardrobe_r as W
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
