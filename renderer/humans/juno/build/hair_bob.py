"""ASSET3 — Juno's v7 groom: a voluminous side-parted wavy BOB (judge r4: the slick bun read as a helmet/cap).

Design (lead r4 decision, original): deep side part on her LEFT (+x), lifted root at the part, the big side sweeps over the
crown and down across the upper forehead as a soft swept fringe that falls beside her right temple (never over the eyes),
the small side is TUCKED BEHIND the left ear, chin-length soft waves, ends curling in. No simulation: rigid with the head
bone, every tip ends above the collar (nape ends >= 4 cm above the neck base).

Construction (all procedural from the morphed base):
  1. roots on the scalp (HeadSurf, soft hairline density taper)
  2. MARCH: every strand is integrated through a flow field (away from the part on top, radial from the crown whorl at the
     back) + gravity, with collision against the head shell at a height that grows with the distance travelled (strands
     rooted near the part lie on top = natural layering + volume), the tuck-behind-ear and face-clearance constraints,
     ends curl in under the bob line
  3. CLUMP HIERARCHY: N_LOCKS big locks (k-means on where the strands GO: mid + tip) -> N_SUB mid clumps per lock ->
     strands; strands converge to their sub-clump guide, sub-guides converge to the lock guide (tapered bundles with gaps
     between locks = depth); each lock has its own wave phase/amplitude (a 3-D wave: side + normal), lift and length
  4. frizz per strand, flyaways leaving the outer layer, baby hairs at the hairline
"""
from __future__ import annotations

import numpy as np
from scipy.cluster.vq import kmeans2
from scipy.spatial import cKDTree

from hair import HeadSurf, nrm, resample, smoothstep

P_BOB = 32
N_LOCKS = 34
N_SUB = 6
PART_X = -0.030         # part line x offset from the skull centre (her left = +x); ASSET3: big side + fringe on her LEFT
                        # (screen-right, the fill side): on the key side the fringe shadowed her eyes (blind check b5)
TUCK_SIDE = -1.0        # -1: her right side (the small side) is tucked behind the ear
DS = 0.0022             # march step (m)
NSTEP = 230


class GridSD:
    """the HeadSurf signed distance + normal sampled once on a 2.5 mm grid (k-NN per step was ~1 s per query under load);
    trilinear lookups. Same sign convention as HeadSurf.sd."""

    def __init__(self, hs, h=0.003, pad=0.06):
        from scipy.ndimage import map_coordinates
        self.mc = map_coordinates
        lo, hi = hs.P.min(0) - pad, hs.P.max(0) + pad
        self.lo, self.h = lo, h
        shp = np.ceil((hi - lo) / h).astype(int) + 1
        g = [lo[i] + h * np.arange(shp[i]) for i in range(3)]
        Q = np.stack(np.meshgrid(*g, indexing="ij"), -1).reshape(-1, 3)
        _, i = hs.kd.query(Q, k=4, workers=16)
        n = nrm(hs.N[i].mean(1))
        s = ((Q[:, None, :] - hs.P[i]) * hs.N[i]).sum(2).mean(1)
        self.S = s.reshape(shp)
        self.N = n.reshape(tuple(shp) + (3,))

    def sd(self, X):
        sh = X.shape
        c = ((X.reshape(-1, 3) - self.lo) / self.h).T
        s = self.mc(self.S, c, order=1, mode="nearest")
        n = np.stack([self.mc(self.N[..., k], c, order=1, mode="nearest") for k in range(3)], 1)
        return s.reshape(sh[:-1]), nrm(n).reshape(sh)

    def push(self, X, h, iters=3):
        X = X.copy()
        h = np.broadcast_to(np.asarray(h, float), X.shape[:-1])
        for _ in range(iters):
            s, n = self.sd(X)
            m = s < h
            if not m.any():
                break
            X[m] += (h[m] - s[m])[:, None] * n[m]
        return X


def _end_y(base, az):
    """bob line (y of the tips) by azimuth (deg, 0 front, +-180 back): chin level at the sides, a touch longer toward
    the front (soft A-line), shorter at the nape (collar clearance)."""
    chin = base.lm["chin"][1]
    a = np.abs(az)
    return chin + np.interp(a, [0, 60, 100, 140, 180], [0.004, 0.006, 0.012, 0.022, 0.030])


class Flow:
    def __init__(self, base, hs):
        self.base, self.hs = base, hs
        c = hs.c
        self.c = c
        self.xp = c[0] + PART_X
        self.zc = c[2] - 0.040                       # crown whorl z (the part runs from the front to here)
        top = hs.P[np.argmax(hs.P[:, 1] - 2.0 * np.abs(hs.P[:, 2] - self.zc) - 2.0 * np.abs(hs.P[:, 0] - self.xp))]
        self.W = top
        e = base.lm
        self.eye_y = e["eye_c_l"][1]
        er = e["eye_r_l"]
        self.brow_y = self.eye_y + 1.95 * er
        self.eye_x = e["eye_c_l"][0] + 1.25 * er
        self.chin_y = e["chin"][1]
        import juno_design
        ew = juno_design.ear_weight(base)
        E = base.VA[ew > 0.5]
        self.ear_lo, self.ear_hi = E.min(0), E.max(0)
        self.gs = GridSD(hs)

    def desired(self, p, n, prog, free=False):
        """unit flow direction at points p (S,3) with head normals n; prog (S,) 0..1 = how far down toward the bob line."""
        c = self.c
        x, y, z = p[:, 0], p[:, 1], p[:, 2]
        side = np.where(x >= self.xp, 1.0, -1.0)
        # top/front zone: away from the part; strength fades with height below the skull centre
        A = 2.6 * smoothstep(c[1] - 0.045, c[1] + 0.03, y) * (0.0 if free else 1.0)
        f_top = np.stack([side, np.zeros_like(x), np.zeros_like(x)], 1)
        # big side, front: the fringe sweeps a little forward/down across the upper forehead
        frontw = smoothstep(c[2] + 0.05, c[2] + 0.10, z) * (side == -TUCK_SIDE)
        f_top = f_top + np.stack([0 * x, -0.15 * frontw, 0.25 * frontw], 1)
        # back zone: radial from the crown whorl
        r = p - self.W
        r[:, 1] = np.minimum(r[:, 1], 0.0)
        f_back = nrm(r)
        wb = smoothstep(self.zc + 0.025, self.zc - 0.025, z)
        f = nrm(f_top * (1 - wb[:, None]) + f_back * wb[:, None]) * A[:, None]
        g = np.array([0.0, -1.0, 0.0])
        d = f + g[None]
        # ends curl in: toward the neck axis, under the bob line
        ax = np.array([0.0, 0.0, c[2] - 0.015])
        inw = np.stack([ax[0] - x, 0 * x, ax[2] - z], 1)
        inw = nrm(inw)
        d = d + inw * (1.1 * smoothstep(0.78, 1.0, prog))[:, None]
        # tangent plane (only pulls along the surface while touching; free hanging below)
        return nrm(d)

    def constrain(self, p, side_of_root, ramp=None):
        """tuck behind the ear (her left) + keep the face clear. In place."""
        x, y, z = p[:, 0], p[:, 1], p[:, 2]
        lo, hi = self.ear_lo, self.ear_hi
        # tuck: on the tuck side, around ear height, nothing in front of the back of the ear root
        tuck = (TUCK_SIDE * x > 0.045) & (side_of_root == TUCK_SIDE)
        wy = smoothstep(lo[1] - 0.035, lo[1] - 0.005, y) * smoothstep(hi[1] + 0.006, hi[1] - 0.008, y)
        zmax = lo[2] - 0.010
        if ramp is not None:
            wy = wy * ramp
        m = tuck & (z > zmax)
        p[m, 2] = z[m] - (z[m] - zmax) * wy[m]
        if m.any() and hasattr(self, "gs"):
            q = p[m]
            s_, n_ = self.gs.sd(q)
            p[m] = q + np.maximum(0.004 - s_, 0)[:, None] * n_
        # face clearance: in front of the face plane, between the chin and the brow, |x| >= xf(y)
        front = z > self.c[2] - 0.005
        xf = np.interp(y, [self.chin_y - 0.01, self.chin_y + 0.03, self.eye_y - 0.03, self.eye_y, self.brow_y],
                       [0.062, 0.078, 0.095, self.eye_x + 0.020, self.eye_x + 0.006])
        band = (y > self.chin_y - 0.03) & (y < self.brow_y + 0.002) & front
        ax = np.abs(x)
        m = band & (ax < xf)
        p[m, 0] = np.sign(x[m] + 1e-9) * xf[m]
        return p


def march(base, hs, fl, roots, layer, rng):
    gs = fl.gs
    S = len(roots)
    s0, n0 = hs.sd(roots)
    rel = roots - fl.c
    az = np.degrees(np.arctan2(rel[:, 0], rel[:, 2]))
    yend = _end_y(base, az) + rng.normal(0, 0.004, S) - 0.004 * (1 - layer)
    side_root = np.where(roots[:, 0] >= fl.xp, 1.0, -1.0)
    p = roots.copy()
    # initial direction: lift off the scalp (strongly at the part) + the flow
    dpart = np.abs(roots[:, 0] - fl.xp)
    lift0 = 0.6 + 1.6 * smoothstep(0.03, 0.0, dpart) * (roots[:, 2] > fl.zc)
    prog0 = np.zeros(S)
    d = nrm(fl.desired(p, n0, prog0) + n0 * lift0[:, None])
    X = np.zeros((S, NSTEP + 1, 3))
    X[:, 0] = p
    alive = np.ones(S, bool)
    dtrav = np.zeros(S)
    # max volume (outer surface height) by region: fuller on top/crown, tighter at the nape
    ytop = fl.c[1]
    for k in range(1, NSTEP + 1):
        idx = np.where(alive)[0]
        if not len(idx):
            X[:, k:] = X[:, k - 1:k]
            break
        pa, da = p[idx], d[idx]
        s, n = gs.sd(pa)
        y0 = roots[idx, 1]
        prog = np.clip((y0 - pa[:, 1]) / np.maximum(y0 - yend[idx], 0.02), 0, 1)
        g = fl.desired(pa, n, prog)
        # comb pressure: while over the upper head the strand hugs the shell (flow projected on the tangent plane +
        # a press toward the skin); once it leaves the shell (below the widest point) only gravity + the curl act
        touch = smoothstep(0.032, 0.024, s) * smoothstep(fl.c[1] - 0.06, fl.c[1] - 0.01, pa[:, 1])
        gt = nrm(g - n * (g * n).sum(1, keepdims=True)) - 0.2 * n
        # free (off the shell): gravity + the end curl only -- no sideways flow (that flared the bob into a bell)
        gfree = fl.desired(pa, n, prog, free=True)
        g = nrm(gfree * (1 - touch[:, None]) + nrm(gt) * touch[:, None])
        beta = 0.30
        dn = nrm(da * (1 - beta) + g * beta)
        pn = pa + dn * DS
        Hm = (0.009 + 0.015 * smoothstep(ytop - 0.06, ytop + 0.07, pn[:, 1])) * (0.45 + 0.55 * layer[idx])
        # big side, front: the swept fringe rolls up off the forehead (a lifted wave behind the hairline)
        fw = smoothstep(fl.c[2] + 0.04, fl.c[2] + 0.09, roots[idx, 2]) * (np.sign(roots[idx, 0] - fl.xp) == -TUCK_SIDE)
        Hm = Hm + 0.008 * fw * np.exp(-((dtrav[idx] - 0.035) / 0.03) ** 2)
        Hm = Hm + 0.009 * smoothstep(0.03, 0.0, np.abs(roots[idx, 0] - fl.xp)) * (roots[idx, 2] > fl.zc)   # part lift
        h = 0.0006 + Hm * (1 - np.exp(-(dtrav[idx] + DS) / 0.03))
        s2, n2 = gs.sd(pn)
        m = s2 < h
        pn[m] += (h[m] - s2[m])[:, None] * n2[m]
        pn = fl.constrain(pn, side_root[idx], smoothstep(0.008, 0.025, dtrav[idx]))
        dn = nrm(pn - pa)
        dtrav[idx] += np.linalg.norm(pn - pa, axis=1)
        p[idx], d[idx] = pn, dn
        X[idx, k] = pn
        dead = np.setdiff1d(np.arange(S), idx)
        X[dead, k] = X[dead, k - 1]
        stop = (pn[:, 1] < yend[idx]) & (prog > 0.99)
        alive[idx[stop]] = False
    return X, dtrav


def _frames(G, hs):
    """per point: tangent, outward (head normal projected), side (S,P,3)."""
    T = nrm(np.gradient(G, axis=1))
    _, n = hs.gs.sd(G) if hasattr(hs, 'gs') else hs.sd(G)
    n = nrm(n - T * (n * T).sum(-1, keepdims=True))
    sd = nrm(np.cross(T, n))
    return T, n, sd


def bob_groom(base, hs, rng, n_strands):
    fl = Flow(base, hs)
    hs.gs = fl.gs
    cand = np.where(hs.M > 0.12)[0]
    wgt = smoothstep(0.12, 0.85, hs.M[cand]) ** 1.5 + 0.02
    ri = rng.choice(cand, n_strands, p=wgt / wgt.sum(), replace=len(cand) < n_strands)
    roots = hs.shell(hs.P[ri] + rng.normal(0, 0.0003, (n_strands, 3)), 0.0)
    layer = rng.random(n_strands)
    X, dtrav = march(base, hs, fl, roots, layer, rng)
    X = resample(X, P_BOB)
    t = np.linspace(0, 1, P_BOB)
    # ---- lock hierarchy (clustered on where the strands go)
    feat = np.concatenate([X[:, P_BOB // 3], X[:, 2 * P_BOB // 3], 0.7 * X[:, -1]], 1)
    sub_i = rng.choice(n_strands, min(12000, n_strands), replace=False)
    cen, _ = kmeans2(feat[sub_i], N_LOCKS, minit="++", seed=4430, iter=25)
    lock = cKDTree(cen).query(feat)[1]
    sub = np.zeros(n_strands, int)
    fs = np.concatenate([X[:, P_BOB // 2], X[:, -1]], 1)
    for k in range(N_LOCKS):
        mk = np.where(lock == k)[0]
        if len(mk) < N_SUB * 4:
            continue
        c2, lab = kmeans2(fs[mk], N_SUB, minit="++", seed=4430 + k, iter=15)
        sub[mk] = lab
    cl = lock * N_SUB + sub
    G_lock = np.zeros((N_LOCKS, P_BOB, 3))
    for k in range(N_LOCKS):
        mk = lock == k
        if mk.any():
            G_lock[k] = X[mk].mean(0)
    G_sub = np.zeros((N_LOCKS * N_SUB, P_BOB, 3))
    for k in np.unique(cl):
        G_sub[k] = X[cl == k].mean(0)
    # converge: strand -> sub guide (strong, late), sub guide -> lock guide (moderate)
    c1 = (0.45 * smoothstep(0.12, 0.75, t))[None, :, None]
    c2 = (0.72 * smoothstep(0.08, 0.6, t))[None, :, None] * rng.uniform(0.75, 1.05, (n_strands, 1, 1))
    Gs = G_sub[cl] + c1 * (G_lock[lock] - G_sub[cl])
    Xc = Gs + (X - G_sub[cl]) * (1 - c2)
    # ---- per-lock wave (3-D), lift and phase; amplitude grows down the strand (top stays smooth)
    T, Nn, Sd = _frames(G_lock, hs)
    arc = np.concatenate([np.zeros((N_LOCKS, 1)), np.cumsum(np.linalg.norm(np.diff(G_lock, axis=1), axis=2), 1)], 1)
    wl = rng.uniform(0.075, 0.10, N_LOCKS)
    ph = rng.random(N_LOCKS) * 6.283
    amp = rng.uniform(0.0025, 0.0045, N_LOCKS)
    ramp = smoothstep(0.05, 0.13, arc)
    wv = np.sin(2 * np.pi * arc / wl[:, None] + ph[:, None]) * amp[:, None] * ramp
    wn = np.cos(2 * np.pi * arc / wl[:, None] + ph[:, None]) * 0.45 * amp[:, None] * ramp
    lift = rng.uniform(0.0, 0.0035, N_LOCKS)[:, None] * smoothstep(0.0, 0.25, t)[None]
    D = Sd * wv[..., None] + Nn * (wn + lift)[..., None]
    # sub clumps carry a little of their own wave on top (breaks the lock into strands-of-strands)
    nsub = N_LOCKS * N_SUB
    phs = rng.random(nsub) * 6.283
    ampS = rng.uniform(0.0005, 0.0013, nsub)
    Ts, Ns, Ss = _frames(G_sub, hs)
    arcS = np.concatenate([np.zeros((nsub, 1)), np.cumsum(np.linalg.norm(np.diff(G_sub, axis=1), axis=2), 1)], 1)
    wS = np.sin(2 * np.pi * arcS / 0.04 + phs[:, None]) * ampS[:, None] * smoothstep(0.03, 0.1, arcS)
    Xc = Xc + D[lock] + (Ss * wS[..., None])[cl]
    # ---- frizz (per strand, grows toward the tip) + stray ends
    fz = rng.normal(0, 1, (n_strands, 1, 3)) * 0.0003 * t[None, :, None] ** 1.5
    fz2 = np.sin(t[None] * rng.uniform(8, 16, (n_strands, 1)) + rng.random((n_strands, 1)) * 6.28)[..., None] \
        * rng.normal(0, 0.0002, (n_strands, 1, 3))
    Xc = Xc + fz + fz2
    Xc[:, 0] = roots
    # ---- collision + constraints again (waves must not enter the head or the face)
    side_root = np.where(roots[:, 0] >= fl.xp, 1.0, -1.0)
    # face band: a strand that would enter the face clearance is CUT where it enters (clamping drew straight lines
    # along the band edge + its hair shadowed the eye); its tip then ends at the temple
    Cf = fl.constrain(Xc.reshape(-1, 3).copy(), np.zeros(n_strands * P_BOB)).reshape(Xc.shape)
    bad = np.linalg.norm(Cf - Xc, axis=2) > 1e-5
    first = np.where(bad.any(1), bad.argmax(1), P_BOB)
    cut = np.clip((first - 1) / (P_BOB - 1) - np.where(first < P_BOB, rng.uniform(0.0, 0.18, n_strands), 0), 0.2, 1.0)   # staggered ends
    Xf = Xc.reshape(-1, 3)
    arcl = np.concatenate([np.zeros((n_strands, 1)), np.cumsum(np.linalg.norm(np.diff(Xc, axis=1), axis=2), 1)], 1)
    Xf = fl.constrain(Xf, np.repeat(side_root, P_BOB), smoothstep(0.008, 0.025, arcl).ravel())
    Xc = Xf.reshape(Xc.shape)
    Xc = fl.gs.push(Xc, 0.0006 + 0.003 * smoothstep(0.0, 0.3, t)[None])
    Xc[:, 0] = roots
    # length jitter: ragged tips (silhouette breakup), inner layer a little shorter
    keep = np.minimum(np.clip(rng.uniform(0.86, 1.0, n_strands) - 0.04 * (1 - layer), 0.6, 1.0), cut)
    Xc = resample(Xc, P_BOB, keep)
    return Xc, cl, layer, fl


def bob_fuzz(base, hs, fl, rng, scalpX, layer, n):
    """baby hairs along the hairline (~60 %) + flyaways leaving the outer layer (~40 %, ~300 of them long)."""
    P = 12
    nb = int(n * 0.6)
    cand = np.where((hs.M > 0.4) & (hs.M < 0.8))[0]
    ri = rng.choice(cand, nb)
    r = hs.shell(hs.P[ri], 0.0)
    _, nn = hs.sd(r)
    g = fl.desired(r, nn, np.zeros(nb))
    g = nrm(g - nn * (g * nn).sum(1, keepdims=True))
    side = np.cross(nn, g)
    L = rng.uniform(0.004, 0.012, nb)
    t = np.linspace(0, 1, P)[None, :, None]
    curl = rng.normal(0, 1.3, nb)
    d = g[:, None] * np.cos(curl[:, None, None] * t) + side[:, None] * np.sin(curl[:, None, None] * t)
    Xb = r[:, None] + np.cumsum(d * (L / (P - 1))[:, None, None], 1) - d[:, :1] * (L / (P - 1))[:, None, None]
    Xb = hs.push(Xb, 0.0004 + 0.0014 * t[..., 0])
    Xb[:, 0] = r
    nf = n - nb
    outer = np.where(layer > 0.7)[0]
    si = rng.choice(outer, nf)
    PS = scalpX.shape[1]
    k0 = rng.integers(PS // 5, PS - 3, nf)
    p0 = scalpX[si, k0]
    tg = nrm(scalpX[si, k0 + 1] - scalpX[si, k0 - 1])
    _, nn = hs.sd(p0)
    ang = rng.uniform(0.15, 0.6, nf)[:, None]
    d0 = nrm(tg * np.cos(ang) + nn * np.sin(ang) + rng.normal(0, 0.3, (nf, 3)))
    bend = nrm(np.cross(d0, rng.normal(0, 1, (nf, 3))))
    L = np.where(rng.random(nf) < 0.2, rng.uniform(0.025, 0.05, nf), rng.uniform(0.008, 0.022, nf))
    bt = rng.uniform(0.6, 2.4, nf)[:, None, None] * t
    d = d0[:, None] * np.cos(bt) + bend[:, None] * np.sin(bt)
    Xf = p0[:, None] + np.cumsum(d * (L / (P - 1))[:, None, None], 1) - d[:, :1] * (L / (P - 1))[:, None, None]
    Xf = hs.push(Xf, 0.002)
    C = fl.constrain(Xf.reshape(-1, 3).copy(), np.zeros(nf * P)).reshape(Xf.shape)
    ok = np.abs(C - Xf).max((1, 2)) < 1e-6          # a flyaway that would cross the face is dropped, never clamped
    return np.concatenate([Xb, Xf[ok]])
