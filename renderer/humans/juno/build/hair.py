"""asset — hair strands: scalp groom (slick back into a low bun, per concept sheet_v4_0), brows, lashes.

build(base, S) -> {group: {"meta": {...}, "lod0": arrays, "lod1": arrays}}
  meta   = hair_group json WITHOUT "lod"/"lod_strands"/"strands"/"points" (build_juno fills those), e.g.
           {"bone":"head", "follow":"bone"|"surface", "surface_submesh":"head", "width_root_m":..,"width_tip_m":..,
            "color_root":[r,g,b], "color_tip":[r,g,b], "melanin":0.8, "redness":0.5, ...any extra shading hints}
  arrays = {"pos": (S,P,3) float rest positions in metres (root first, every strand same P),
            "width": (S,2) root/tip width m, "seed": (S,) 0..1, "clump": (S,) int}
build_juno binds every strand root to the nearest triangle of `surface_submesh` (default "head") -> tri/bary + root_uv
and writes hair/lod0_<group>.bin / lod1_<group>.bin. Groups: scalp, brows_l, brows_r, lashes_upper_l, lashes_upper_r,
lashes_lower_l, lashes_lower_r, fuzz. lod1 total <= 15k strands.
juno_design.scalp_mask(base) gives the agreed hairline (0..1 per MH vertex); skin_tex paints the scalp under it.

Everything is procedural from the morphed base (landmarks, skull_center, scalp_mask, MH helper eyelash strips):
  HeadSurf      dense samples of the head skin -> signed distance / shell projection / collision push-out
  scalp groom   every root is combed along the great circle (about the skull centre) toward the low bun's tie ring;
                height above the skin grows with the distance travelled (hairline strands lie on top = slick layering),
                clumps pull strands onto guide paths late (comb lines), then the strand continues into one of K rope
                coils wound around an oblate bun on the back of the head (twisted cross-section, tips tucked)
  fuzz          baby hairs at the hairline + flyaways leaving the outer layer and the bun
  brows         2D brow shape in eye-radius units projected onto the skin, strands lying on the skin
  lashes        roots on the skin lid margin (juno_sculpt.lid_margins), so surface follow rides the blink
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

import juno_design

RNG_SEED = 4430
P_SCALP, P_FUZZ, P_BROW, P_LASH = 24, 12, 8, 10
N_LOD0 = {"scalp": 56000, "fuzz": 4200, "brow": 1400, "lash_up": 200, "lash_lo": 24}
N_LOD1 = {"scalp": 11000, "fuzz": 500, "brow": 460, "lash_up": 130, "lash_lo": 20}

TENDRILS = False
STYLE = "bob"     # ASSET3 v7: "bob" (hair_bob: side-parted wavy bob, tucked left) | "bun" (v1-v6 slick low bun)
N_BIG = 12        # ASSET2 v6: big locks in the groom hierarchy
HAIR_ROOT = [0.062, 0.034, 0.019]   # linear RGB, dark chestnut (ASSET2 v5: lifted off black)
HAIR_TIP = [0.13, 0.072, 0.040]   # ASSET3 v7: darker tips (long bob strands read blonde-ish)


def smoothstep(a, b, x):
    t = np.clip((np.asarray(x, float) - a) / (b - a), 0, 1)
    return t * t * (3 - 2 * t)


def nrm(v):
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-12)


def resample(X, P, keep=None):
    """X (S,n,3) polylines -> (S,P,3) uniform in arc length. keep (S,) in (0,1]: keep only that leading fraction."""
    S, n, _ = X.shape
    seg = np.linalg.norm(np.diff(X, axis=1), axis=2)
    L = np.concatenate([np.zeros((S, 1)), np.cumsum(seg, 1)], 1)
    tot = L[:, -1:] * (1.0 if keep is None else np.asarray(keep)[:, None])
    u = np.linspace(0, 1, P)[None] * tot
    off = (np.arange(S) * (L[:, -1].max() + 1.0))[:, None]
    flatL = (L + off).ravel()
    q = (u + off).ravel()
    j = np.clip(np.searchsorted(flatL, q, side="right") - 1, 0, n * S - 2)
    j = np.minimum(j, (np.arange(S)[:, None] * n + n - 2).repeat(P, 1).ravel())
    l0, l1 = flatL[j], flatL[j + 1]
    w = np.clip((q - l0) / np.maximum(l1 - l0, 1e-12), 0, 1)[:, None]
    Xf = X.reshape(-1, 3)
    return (Xf[j] * (1 - w) + Xf[j + 1] * w).reshape(S, P, 3)


class HeadSurf:
    """Dense random samples of the head skin with smooth normals + hairline mask; k-NN signed distance."""

    def __init__(self, base, spacing=0.0008, rng=None):
        rng = rng or np.random.default_rng(RNG_SEED)
        c = juno_design.skull_center(base)
        self.c = c
        mask = juno_design.scalp_mask(base)
        tri = []
        for fi in base.faces_of("body"):
            vs = [vi for vi, _ in base.obj.faces[fi]]
            tri += [[vs[0], vs[1], vs[2]]] + ([[vs[0], vs[2], vs[3]]] if len(vs) == 4 else [])
        tri = np.array(tri)
        V = base.VA
        neck_y = base.lm["neck"][1]
        keep = (np.linalg.norm(V[tri] - c, axis=2).max(1) < 0.2) & (V[tri][:, :, 1].min(1) > neck_y - 0.03)
        tri = tri[keep]
        self.tri = tri
        a, b, cc = V[tri[:, 0]], V[tri[:, 1]], V[tri[:, 2]]
        area = 0.5 * np.linalg.norm(np.cross(b - a, cc - a), axis=1)
        M = int(area.sum() / spacing ** 2)
        t = rng.choice(len(tri), M, p=area / area.sum())
        r1, r2 = rng.random(M), rng.random(M)
        s1 = np.sqrt(r1)
        bc = np.stack([1 - s1, s1 * (1 - r2), s1 * r2], 1)
        T = tri[t]
        self.P = (bc[:, :, None] * V[T]).sum(1)
        self.N = nrm((bc[:, :, None] * base.N[T]).sum(1))
        # evaluate the agreed hairline function at the samples themselves (smooth edge, no per-triangle steps);
        # outside the scalp region keep the interpolated vertex mask (it carries the skin-only / radius cut)
        Mv = (bc * mask[T]).sum(1)

        class _Pts:
            pass
        sh = _Pts()
        sh.VA, sh.lm = self.P, base.lm
        sh.skin_mask = np.ones(len(self.P), bool)
        Mp = juno_design.scalp_mask(sh, soft_deg=9.0)   # ASSET2 v6: wider hairline density taper
        # irregular, tapered hairline (lead v2 review: no straight helmet edge): low-frequency positional noise
        # shifts the edge by a few mm along it, only inside the transition band
        Q = self.P * 160.0
        nz = (np.sin(Q[:, 0] * 1.7 + 0.3) * np.sin(Q[:, 1] * 1.3 + 1.1) + 0.6 * np.sin(Q[:, 0] * 4.1 + Q[:, 2] * 3.3)
              + 0.4 * np.sin(Q[:, 1] * 5.7 - Q[:, 0] * 2.9 + 2.0)) / 2.0
        band = 4 * Mp * (1 - Mp)
        Mp = np.clip(Mp + 0.35 * nz * band, 0, 1)
        self.M = np.where((mask[T] > 0).any(1), Mp, Mv)
        self.M = self.M * (1 - (bc * juno_design.ear_weight(base)[T]).sum(1))   # ASSET2 v6: never root on the ear
        self.kd = cKDTree(self.P)
        d = self.P - c
        r = np.linalg.norm(d, axis=1)
        self.elev = np.degrees(np.arcsin(np.clip(d[:, 1] / r, -1, 1)))
        self.az = np.degrees(np.arctan2(d[:, 0], d[:, 2]))          # signed: + = her left
        hl = self.P[(self.M > 0.3) & (self.M < 0.7)]
        self.hl_kd = cKDTree(hl)

    def sd(self, X, k=4):
        sh = X.shape
        Xf = X.reshape(-1, 3)
        _, i = self.kd.query(Xf, k=k)
        n = nrm(self.N[i].mean(1))
        s = ((Xf[:, None, :] - self.P[i]) * self.N[i]).sum(2).mean(1)
        return s.reshape(sh[:-1]), n.reshape(sh)

    def shell(self, X, h):
        s, n = self.sd(X)
        return X + (np.asarray(h) - s)[..., None] * n

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

    def hairline_dist(self, X):
        d, _ = self.hl_kd.query(X.reshape(-1, 3))
        return d.reshape(X.shape[:-1])


# ------------------------------------------------------------------------------------------------ scalp groom
BUN_ELEV, BUN_AZ = 13.0, 180.0      # bun direction from the skull centre (deg): low bun at the back, ~eye height


class Bun:
    """low bun: oblate spheroid sitting on the back of the skull; ax = scalp normal there (pointing back)."""

    def __init__(self, hs, base):
        c = hs.c
        el, az = np.radians(BUN_ELEV), np.radians(BUN_AZ)
        bdir = np.array([np.cos(el) * np.sin(az), np.sin(el), np.cos(el) * np.cos(az)])
        u = nrm(hs.P - c)
        cand = np.where(hs.M > 0.9)[0]
        i = cand[np.argmax(u[cand] @ bdir)]
        self.Bb = hs.P[i]
        self.Hs = float(np.linalg.norm(self.Bb - c))
        k = self.Hs / 0.092
        self.ax = nrm(0.6 * hs.N[i] + 0.4 * bdir + np.array([0, 0.12, 0]))
        self.e1 = nrm(np.cross([0, 1.0, 0], self.ax))                # horizontal
        self.e2 = np.cross(self.ax, self.e1)                          # ~ up
        self.Rb = 0.045 * k   # ASSET2 v5: bigger bun, reads in 3/4 + profile
        self.Rh = 0.036 * k
        self.C = self.Bb + self.ax * self.Rh * 0.92
        self.r_tie = 0.016 * k
        self.bhat = nrm(self.Bb - c)
        self.th_tie = self.r_tie / self.Hs

    def surf(self, th, ph, sc):
        """ellipsoid point at azimuth th, elevation ph, radial scale sc (arrays broadcast) -> (...,3) + outward dir"""
        th, ph, sc = np.broadcast_arrays(th, ph, sc)
        cp = np.cos(ph)[..., None]
        rad = np.cos(th)[..., None] * self.e1 + np.sin(th)[..., None] * self.e2
        p = self.C + sc[..., None] * (self.Rb * cp * rad + self.Rh * np.sin(ph)[..., None] * self.ax)
        out = nrm(cp * rad / self.Rb + np.sin(ph)[..., None] * self.ax / self.Rh)
        return p, out, rad

    def azimuth(self, X):
        d = X - self.Bb
        return np.arctan2(d @ self.e2, d @ self.e1)


def thickness(hs, bun, X):
    """slick groom volume (m) above the skin at X: thin at the hairline, a little fuller over the crown, gathering
    toward the tie of the bun."""
    dh = hs.hairline_dist(X)
    d = X - hs.c
    r = np.linalg.norm(d, axis=-1)
    elev = np.degrees(np.arcsin(np.clip(d[..., 1] / r, -1, 1)))
    crown = smoothstep(5, 50, elev)
    db = np.linalg.norm(X - bun.Bb, axis=-1)
    # ASSET2 v5: real volume (lead: 8-14 mm over the crown, soft lift at the hairline, fuller toward the bun)
    V = (0.0045 + 0.0075 * crown) * (0.5 + 0.5 * smoothstep(0.0, 0.025, dh))
    V += 0.006 * smoothstep(0.06, 0.015, db)
    return V * (bun.Hs / 0.092)


def slerp_paths(c, P0, bhat, n, th_tie):
    """great-circle paths (directions from c) from P0 toward bhat, stopping th_tie short. -> (S,n,3) unit dirs, (S,) angle"""
    u0 = nrm(P0 - c)
    cosw = np.clip(u0 @ bhat, -1, 1)
    w = np.arccos(cosw)
    tgt = np.maximum(w - th_tie, 0.0)
    perp = nrm(bhat[None] - cosw[:, None] * u0 + 1e-9 * np.array([0, 1.0, 0]))
    t = np.linspace(0, 1, n)[None]
    ang = (tgt[:, None] * t)[..., None]
    return np.cos(ang) * u0[:, None] + np.sin(ang) * perp[:, None], tgt


def march_to_bun(hs, bun, roots, layer, n=32):
    """roots (S,3) -> scalp paths (S,n,3) combed straight back (great circles from the skull centre) to the tie ring.
    Height above the skin grows with the distance travelled from the root, so strands rooted further away lie on
    top (slick layering: the hairline strands cover everything behind them)."""
    U, tgt = slerp_paths(hs.c, roots, bun.bhat, n, bun.th_tie)
    X = hs.c + U * np.linalg.norm(roots - hs.c, axis=1)[:, None, None]
    X = hs.shell(X, 0.0)
    X[:, 0] = roots
    seg = np.linalg.norm(np.diff(X, axis=1), axis=2)
    dtrav = np.concatenate([np.zeros((len(X), 1)), np.cumsum(seg, 1)], 1)
    lift = 1 - np.exp(-dtrav / (0.035 * bun.Hs / 0.092))
    h = 0.0005 + thickness(hs, bun, X) * np.clip(lift * (0.55 + 0.45 * layer[:, None]), 0, 1)
    X = hs.shell(X, h)
    X[:, 0] = roots
    return X, h


def scalp_groom(base, hs, bun, rng, n_strands):
    # roots: skin samples under the hair, density fading in over the soft hairline
    cand = np.where(hs.M > 0.12)[0]
    wgt = smoothstep(0.12, 0.85, hs.M[cand]) ** 1.5 + 0.02     # density tapers over the hairline band
    ri = rng.choice(cand, n_strands, p=wgt / wgt.sum(), replace=len(cand) < n_strands)
    roots = hs.P[ri] + rng.normal(0, 0.0003, (n_strands, 3))
    roots = hs.shell(roots, 0.0)
    # guides = clump centres: combed ribbons, elongated along the flow (cluster in (across-flow, along-flow*0.35))
    area = (hs.M > 0.5).sum() * 0.0008 ** 2
    G = int(area / 1.3e-4)   # ASSET2 v5/v6: fewer, bigger clumps = readable comb lines
    gi = rng.choice(n_strands, G, replace=False)
    groot = roots[gi]
    cdist, clump = cKDTree(groot).query(roots)
    glayer = rng.random(G)
    layer = np.clip(0.6 * glayer[clump] + 0.4 * rng.random(n_strands), 0, 1)
    GX, gh = march_to_bun(hs, bun, groot, glayer)
    SX, sh = march_to_bun(hs, bun, roots, layer)
    na = SX.shape[1]
    t = np.linspace(0, 1, na)
    # clumping: strands converge onto their guide path late along the strand (comb lines), loosely
    ct = 0.8 * smoothstep(0.08, 0.6, t)[None, :, None] * (0.6 + 0.4 * rng.random((n_strands, 1, 1)))
    Xc = SX * (1 - ct) + GX[clump] * ct
    # frizz: tiny per-strand wobble across the flow
    ph = rng.random((n_strands, 1)) * 6.283
    fr = 0.00022 * np.sin(t[None] * rng.uniform(5, 11, (n_strands, 1)) + ph)
    _, nn = hs.sd(Xc)
    side = nrm(np.cross(np.gradient(Xc, axis=1), nn))
    Xc = Xc + side * fr[..., None]
    # ASSET2 v5: LOCK relief (judge round 1: flat helmet, no clumps/volume): each clump is a lock with its own lift, its
    # centre riding higher than its edges (dark partings between locks), and a slow swept-back wave shared by the lock
    rcl = np.percentile(cdist, 90) + 1e-6
    G_ = len(groot)
    lockH = rng.uniform(0.0008, 0.0050, G_) * (bun.Hs / 0.092)
    prof = 0.35 + 0.65 * (1 - np.clip(cdist / rcl, 0, 1)) ** 1.5
    ramp = smoothstep(0.0, 0.18, t)[None] * (1 - 0.6 * smoothstep(0.75, 1.0, t)[None])
    hlock = (lockH[clump] * prof)[:, None] * ramp
    wph = rng.random(G_) * 6.283
    wamp = rng.uniform(0.0008, 0.0022, G_) * (bun.Hs / 0.092)
    wave = (wamp[clump])[:, None] * np.sin(2 * np.pi * 1.3 * t[None] + wph[clump][:, None]) * ramp
    Xc = Xc + side * wave[..., None]
    # ASSET2 v6: clump HIERARCHY (judge r2): N_BIG big locks sweeping back, seeded by root azimuth/elevation on the scalp
    # (so each is a band along the flow). Strands of a lock converge laterally toward the lock's centre path late along
    # the strand (visible separation + dark partings between locks), each lock has its own lift; the front locks lift
    # into a soft quiff/wave right after the hairline.
    u0 = nrm(roots - hs.c)
    az0 = np.arctan2(u0[:, 0], u0[:, 2])
    el0 = np.arcsin(np.clip(u0[:, 1], -1, 1))
    feat = np.c_[az0 * np.cos(el0), 0.35 * el0]                 # across-flow dominant
    gfe = feat[gi]
    seeds = gfe[rng.choice(len(gfe), N_BIG, replace=False)]
    for _ in range(8):                                          # a few Lloyd steps on the guides
        lab = cKDTree(seeds).query(gfe)[1]
        seeds = np.array([gfe[lab == k].mean(0) if (lab == k).any() else seeds[k] for k in range(N_BIG)])
    dd, big = cKDTree(seeds).query(feat, k=2)
    big = big[:, 0]
    edge = np.clip((dd[:, 1] - dd[:, 0]) / (dd[:, 1] + dd[:, 0] + 1e-9) * 3.0, 0, 1)   # 0 at a parting, 1 inside a lock
    kH = bun.Hs / 0.092
    bigH = rng.uniform(0.0015, 0.0055, N_BIG) * kH
    rampb = smoothstep(0.05, 0.3, t)[None] * (1 - 0.7 * smoothstep(0.7, 1.0, t)[None])
    hbig = (bigH[big] * (0.25 + 0.75 * edge ** 0.7))[:, None] * rampb
    # lateral convergence to the lock's mean path (mean over its strands, index-aligned)
    cen = np.zeros((N_BIG,) + Xc.shape[1:])
    for k in range(N_BIG):
        mk = big == k
        if mk.any():
            cen[k] = Xc[mk].mean(0)
    pull = 0.28 * smoothstep(0.2, 0.75, t)[None, :, None] * (1 - edge[:, None, None] * 0.3)
    dl = (cen[big] - Xc)
    _, nn2 = hs.sd(Xc)
    dl = dl - nn2 * (dl * nn2).sum(-1, keepdims=True)
    Xc = Xc + pull * dl
    # soft front quiff: strands rooted on the front hairline lift a few mm just behind the hairline, then lie back down
    front = smoothstep(0.55, 0.9, u0[:, 2]) * smoothstep(0.2, 0.5, u0[:, 1])
    quiff = (0.0055 * kH * front)[:, None] * (np.sin(np.pi * np.clip(t / 0.45, 0, 1)) ** 1.5)[None]
    hfin = sh * (1 - ct[..., 0]) + gh[clump] * ct[..., 0] + hlock + hbig + quiff
    Xc = hs.shell(Xc, hfin)
    Xc[:, 0] = roots
    # ---- bun: K ropes coiled around the spheroid; each guide (clump) lives in one rope (by its arrival azimuth)
    K = 6
    tie = Xc[:, -1]
    psi_g = bun.azimuth(GX[:, -1])
    rope = np.floor(((psi_g + np.pi) / (2 * np.pi)) * K).astype(int) % K
    r_rope = 0.36 * bun.Rb
    th0 = np.sort(rng.uniform(-np.pi, np.pi, K))
    th0 = -np.pi + (np.arange(K) + 0.5) * 2 * np.pi / K
    turns = rng.uniform(0.85, 1.15, K)
    ph_end = rng.uniform(0.8, 1.2, K)
    rsc = rng.uniform(0.9, 1.03, K)
    bulge_ph = rng.random(K) * 6.283
    ga = np.sqrt(rng.random(G)) * 0.8
    gb = rng.random(G) * 6.283
    sa = ga[clump] + np.sqrt(rng.random(n_strands)) * 0.25
    sb = gb[clump] + rng.normal(0, 0.4, n_strands)
    nb = 48
    s = np.linspace(0, 1, nb)[None]
    k = rope[clump]
    end = rng.uniform(0.85, 1.0, n_strands)                      # strand ends at its own fraction of the rope
    # the rope starts at the strand's own arrival azimuth (no crossing at the tie) and coils around + up the bun
    psi_s = bun.azimuth(tie)
    th = psi_s[:, None] * (1 - smoothstep(0, 0.25, s)) + (th0[k][:, None]) * smoothstep(0, 0.25, s) \
        + 2 * np.pi * turns[k][:, None] * s
    phi = -1.05 + (ph_end[k][:, None] + 1.05) * s ** 0.8
    sc = 0.78 * rsc[k][:, None] * (1 + 0.05 * np.sin(3 * th + bulge_ph[k][:, None])) * (1 - 0.3 * s ** 2)
    C, out, _ = bun.surf(th, phi, sc)
    tang = nrm(np.gradient(C, axis=1))
    bino = nrm(np.cross(tang, out))
    tw = sb[:, None] + 2 * np.pi * 0.6 * s
    rr = r_rope * (1 - 0.4 * s) * sa[:, None]
    off = rr[..., None] * (np.cos(tw)[..., None] * out + np.sin(tw)[..., None] * bino)
    B = C + off
    loose = rng.random(n_strands) < 0.03
    B[loose] += out[loose] * (0.003 * np.sin(s * rng.uniform(3, 8, (loose.sum(), 1)) * 3.14)[..., None] + 0.0015)
    blend = (1 - smoothstep(0.0, 0.3, s))[..., None]
    B = B + (tie[:, None] - B[:, :1]) * blend
    X = np.concatenate([Xc, B[:, 1:]], 1)
    la = np.linalg.norm(np.diff(Xc, axis=1), axis=2).sum(1)
    lb = np.linalg.norm(np.diff(B, axis=1), axis=2).sum(1)
    keep = (la + end * lb) / (la + lb)
    X = resample(X, P_SCALP, keep)
    X = hs.push(X, 0.0006)
    X[:, 0] = roots
    return X, clump, layer


def tendrils(base, hs, bun, rng, n=900):
    """ASSET2 v5 (contract design: two face-framing strands): a soft lock falls from each temple in front of the ear to
    about the jaw line, S-curved, split into a few sub-locks; rigid with the head (never reaches the collar)."""
    c = hs.c
    k = bun.Hs / 0.092
    out, cl = [], []
    for sg in (1.0, -1.0):
        d = hs.P - c
        az = np.degrees(np.arctan2(sg * d[:, 0], d[:, 2]))
        sel = np.where((hs.M > 0.35) & (hs.M < 0.75) & (az > 60) & (az < 80))[0]
        y = hs.P[sel, 1]
        sel = sel[y < np.percentile(y, 35)]
        r0 = hs.P[sel].mean(0)
        nn = nrm(hs.N[sel].mean(0))
        L = 0.076 * k
        nsub = 4
        sub = rng.integers(0, nsub, n)
        sub_off = rng.normal(0, 1, (nsub, 2)) * np.array([0.004, 0.003]) * k
        jit = rng.normal(0, 1, (n, 2)) * 0.0024 * k
        s = np.linspace(0, 1, P_SCALP)[None, :]
        ph = rng.random(nsub)[sub][:, None] * 0.8
        # local frame: down, outward (head normal at the root, horizontal), forward
        out_d = nrm(np.array([nn[0], 0.0, nn[2]]))
        fwd = nrm(np.cross(out_d, [0, 1.0, 0]) * (-sg))
        if fwd[2] < 0:
            fwd = -fwd
        X = (r0[None, None]
             + (-L * s)[..., None] * np.array([0, 1.0, 0])
             + ((0.007 * k * np.sin(np.pi * s) + 0.004 * k * s) + (sub_off[sub, 0][:, None] + jit[:, :1]) * s)[..., None] * out_d
             + ((0.006 * k * np.sin(2 * np.pi * s * 0.9 + ph) - 0.004 * k * s) + (sub_off[sub, 1][:, None] + jit[:, 1:]) * s)[..., None] * fwd)
        X[:, :, :] += rng.normal(0, 0.0005, (n, 1, 3)) * (1 - s)[..., None]
        X = hs.push(X, 0.004)
        keep = rng.uniform(0.75, 1.0, n)
        X = resample(X, P_SCALP, keep)
        out.append(X)
        cl.append(900000 + int(sg > 0) * 10 + sub)
    return np.concatenate(out), np.concatenate(cl)


def fuzz(base, hs, bun, rng, scalpX, layer, n):
    """baby hairs at the hairline (1/3) + flyaways leaving the outer layer of the groom and the bun (2/3)."""
    nb = n // 2   # ASSET2 v6: more baby hairs along the hairline
    cand = np.where((hs.M > 0.45) & (hs.M < 0.8))[0]
    ri = rng.choice(cand, nb)
    r = hs.shell(hs.P[ri], 0.0)
    _, nn = hs.sd(r)
    U, _ = slerp_paths(hs.c, r, bun.bhat, 2, 0.0)
    g = U[:, 1] - U[:, 0]
    g = nrm(g - nn * (g * nn).sum(1, keepdims=True))
    side = np.cross(nn, g)
    L = rng.uniform(0.004, 0.011, nb)
    t = np.linspace(0, 1, P_FUZZ)[None, :, None]
    curl = rng.normal(0, 1.2, nb)
    d = g[:, None] * np.cos(curl[:, None, None] * t) + side[:, None] * np.sin(curl[:, None, None] * t)
    Xb = r[:, None] + np.cumsum(d * (L / (P_FUZZ - 1))[:, None, None], 1) - d[:, :1] * (L / (P_FUZZ - 1))[:, None, None]
    Xb = hs.push(Xb, 0.0004 + 0.0012 * t[..., 0])
    Xb[:, 0] = r
    nf = n - nb
    outer = np.where(layer > 0.75)[0]
    si = rng.choice(outer, nf)
    k0 = rng.integers(4, P_SCALP - 3, nf)
    p0 = scalpX[si, k0]
    tg = nrm(scalpX[si, k0 + 1] - scalpX[si, k0 - 1])
    s0, nn = hs.sd(p0)
    away = nrm(p0 - bun.C)
    onbun = np.linalg.norm(p0 - bun.C, axis=1) < 1.2 * bun.Rb
    nn[onbun] = away[onbun]
    ang = rng.uniform(0.1, 0.45, nf)[:, None]
    d0 = nrm(tg * np.cos(ang) + nn * np.sin(ang) + rng.normal(0, 0.25, (nf, 3)))
    bend = nrm(np.cross(d0, rng.normal(0, 1, (nf, 3))))
    L = np.where(rng.random(nf) < 0.08, rng.uniform(0.025, 0.045, nf), rng.uniform(0.006, 0.022, nf))   # v6: ~200 long flyaways break the silhouette
    bt = rng.uniform(0.6, 2.2, nf)[:, None, None] * t
    d = d0[:, None] * np.cos(bt) + bend[:, None] * np.sin(bt)
    Xf = p0[:, None] + np.cumsum(d * (L / (P_FUZZ - 1))[:, None, None], 1) - d[:, :1] * (L / (P_FUZZ - 1))[:, None, None]
    Xf = hs.push(Xf, 0.002)
    return np.concatenate([Xb, Xf])


# ------------------------------------------------------------------------------------------------ face hair
def raycast_z(V, tris, xy, z0=1.0):
    """first hit of rays from (x,y,z0) toward -z against tris -> (S,3) points, (S,) tri index (-1 = miss)."""
    a, b, c = V[tris[:, 0]], V[tris[:, 1]], V[tris[:, 2]]
    e1, e2 = b - a, c - a
    d = np.array([0, 0, -1.0])
    pv = np.cross(d, e2)
    det = (e1 * pv).sum(1)
    out = np.full((len(xy), 3), np.nan)
    idx = np.full(len(xy), -1)
    for i0 in range(0, len(xy), 256):
        o = np.c_[xy[i0:i0 + 256], np.full(len(xy[i0:i0 + 256]), z0)]
        tv = o[:, None] - a[None]
        u = (tv * pv[None]).sum(2) / det[None]
        qv = np.cross(tv, e1[None])
        v = (qv * d).sum(2) / det[None]
        t = (qv * e2[None]).sum(2) / det[None]
        ok = (u >= 0) & (v >= 0) & (u + v <= 1) & (t > 0) & (np.abs(det)[None] > 1e-14)
        t = np.where(ok, t, np.inf)
        j = t.argmin(1)
        tt = t[np.arange(len(o)), j]
        hit = np.isfinite(tt)
        out[i0:i0 + 256][hit] = o[hit] + tt[hit, None] * d
        idx[i0:i0 + 256][hit] = j[hit]
    return out, idx


def brows(base, hs, rng, side, n):
    ec, er = base.eye[side]
    sg = 1.0 if side == "l" else -1.0
    # centreline in eye units (u: + = lateral, v: + = up), inner head -> arch peak -> tail
    ku = np.array([-1.0, -0.55, 0.0, 0.5, 0.95, 1.35])
    kv = np.array([1.30, 1.45, 1.60, 1.65, 1.52, 1.30]) + 0.17   # lifted ~4-5 mm (lead v2 review)
    kw = np.array([0.08, 0.15, 0.165, 0.14, 0.085, 0.025])   # half-thickness, ~35 % thinner, soft inner head
    # asset-eyes3: density falls off softly at the HEAD (inner end: sparse, feathered) and the tail (oversample,
    # then keep each strand with probability dens(u)); the head also scatters a few strays inward/up into the skin
    u = rng.random(int(n * 1.9)) ** 0.92
    dens = (0.12 + 0.88 * smoothstep(0.0, 0.32, u)) * (1 - 0.45 * smoothstep(0.78, 1.0, u))
    u = u[rng.random(len(u)) < dens][:n]
    n = len(u)
    us = np.interp(u, np.linspace(0, 1, 6), ku)
    vs = np.interp(u, np.linspace(0, 1, 6), kv)
    ws = np.interp(u, np.linspace(0, 1, 6), kw)
    du = np.gradient(np.interp(np.linspace(0, 1, 200), np.linspace(0, 1, 6), ku))
    dv = np.gradient(np.interp(np.linspace(0, 1, 200), np.linspace(0, 1, 6), kv))
    ti = np.clip((u * 199).astype(int), 0, 199)
    tan2 = nrm(np.stack([du[ti], dv[ti]], 1))
    nor2 = np.stack([-tan2[:, 1], tan2[:, 0]], 1)
    # crisp upper edge (uniform density up to +1), soft lower edge (density tails off below -0.6)
    w = 1 - 2 * rng.random(n) ** 0.85
    w = np.where(w < -0.6, -0.6 - (-0.6 - w) * rng.uniform(1.0, 1.8, n), w)
    # head of the brow is rounded/soft: thinner density at the very ends
    head = 1 - smoothstep(0.0, 0.2, u)
    w = w * (1 + 0.5 * head * rng.random(n))                        # feathered: the head spreads wider, thinner
    p2 = np.stack([us, vs], 1) + nor2 * (w * ws)[:, None]
    p2[:, 0] -= head * rng.exponential(0.05, n)                     # strays before the head fade into the skin
    xy = np.c_[ec[0] + sg * p2[:, 0] * er, ec[1] + p2[:, 1] * er]
    near = np.where(np.linalg.norm(base.VA[hs.tri].mean(1) - ec, axis=1) < 4 * er)[0]
    roots, hit = raycast_z(base.VA, hs.tri[near], xy, ec[2] + 0.1)
    ok = hit >= 0
    roots, p2, tan2, nor2, w, u = roots[ok], p2[ok], tan2[ok], nor2[ok], w[ok], u[ok]
    n = len(roots)
    _, nn = hs.sd(roots)
    # direction swings from up (inner head) to along the brow, then down-sideways at the tail; upper-row strands
    # lean in toward the spine (crisp top), lower-row ones lean up (feathered bottom)
    up = np.array([0.2, 1.0])
    mix = smoothstep(0.06, 0.42, u)[:, None]                         # asset-eyes3: longer upward-brushed head
    down = smoothstep(0.62, 1.0, u)[:, None]
    along = nrm(tan2 * (1 - 0.5 * down) + np.array([0.0, -0.5]) * down)
    jit = rng.normal(0, 1, (n, 2)) * (0.14 + 0.16 * head)[:, None]
    d2 = nrm(up * (1 - mix) + along * mix - nor2 * (0.3 * w * mix[:, 0])[:, None] + jit)
    d3 = np.c_[sg * d2[:, 0], d2[:, 1], np.zeros(n)]
    d3 = nrm(d3 - nn * (d3 * nn).sum(1, keepdims=True))
    L = er * rng.uniform(0.28, 0.42, n) * (1 - 0.3 * smoothstep(0.6, 1.0, u)) * (1 - 0.3 * head)
    t = np.linspace(0, 1, P_BROW)
    lift = er * (0.035 + 0.06 * np.sin(np.pi * t * 0.8))[None]
    X = roots[:, None] + d3[:, None] * (L[:, None] * t[None])[..., None] + nn[:, None] * lift[..., None]
    X = hs.push(X, 0.00022)
    X[:, 0] = roots
    return X


def _margin_curve(base, side, upper):
    """skin lid-margin vertices (juno_sculpt.lid_margins) ordered medial -> lateral as a polyline (m,3)."""
    from juno_sculpt import lid_margins
    up, lo = lid_margins(base, side)
    ids = up if upper else lo
    ec, er = base.eye[side]
    V = base.VA[ids]
    sg = 1.0 if side == "l" else -1.0
    o = np.argsort(sg * (V[:, 0] - ec[0]))
    V = V[o]
    # drop vertices that sit far behind the eye front (inner canthus pocket) and smooth the y wiggle
    for _ in range(2):
        V[1:-1] = 0.25 * V[:-2] + 0.5 * V[1:-1] + 0.25 * V[2:]
    return V


LASH_N_UP = 220          # (pre-eyes4 strand count, unused since eyes4: LASH_CLUMPS_UP x LASH_PER_CLUMP)
LASH_CLUMP = 5
# asset-eyes4: lashes are the WHOLE lash fringe now (the lash-line strip is only a faint root shadow): upper =
# LASH_CLUMPS_UP tapered curved clumps of 3-6 strands converging at the tip, dense along the rim, dark BROWN, longest at
# the outer third with an outward sweep (the "wing" comes from them, nothing painted); lower = sparse fine short singles.
LASH_CLUMPS_UP = 56
LASH_PER_CLUMP = (4, 7)
LASH_N_LO = 22
LASH_COL_ROOT = [0.030, 0.018, 0.012]
LASH_COL_TIP = [0.055, 0.034, 0.022]


def _lash_shape(roots, d0, curl_dir, L, curl, ec, er):
    n = len(roots)
    t = np.linspace(0, 1, P_LASH)
    X = np.zeros((n, P_LASH, 3))
    X[:, 0] = roots
    p = roots.copy()
    ds = L / (P_LASH - 1)
    for k in range(1, P_LASH):
        ang = curl * t[k] ** 1.3
        dk = nrm(d0 * np.cos(ang)[:, None] + curl_dir * np.sin(ang)[:, None]) if np.ndim(curl) else \
            nrm(d0 * np.cos(ang) + curl_dir * np.sin(ang))
        p = p + dk * ds[:, None]
        X[:, k] = p
    return X, t


def lashes(base, hs, rng, side, upper, n):
    """roots on the skin lid margin (juno_sculpt.lid_margins), so surface-follow binds them to the margin triangles and
    they ride (and rotate with) the blink. asset-eyes4: upper = clumps (see LASH_CLUMPS_UP), lower = sparse fine.
    -> (X (n,P,3), clump ids (n,))"""
    ec, er = base.eye[side]
    sg = 1.0 if side == "l" else -1.0
    R = _margin_curve(base, side, upper)
    seg = np.linalg.norm(np.diff(R, axis=0), axis=1)
    Lc = np.concatenate([[0], np.cumsum(seg)])
    vert = np.array([0, 1.0 if upper else -1.0, 0])
    lat = np.array([sg, 0, 0])

    def at(u):
        roots = np.stack([np.interp(u * Lc[-1], Lc, R[:, k]) for k in range(3)], 1)
        tg = nrm(np.stack([np.interp(u * Lc[-1], Lc, np.gradient(R[:, k])) for k in range(3)], 1))
        _, nn = hs.sd(roots)
        radial = roots - ec
        radial = nrm(radial - tg * (radial * tg).sum(1, keepdims=True))   # away from the eye centre, across the lid
        return roots, tg, nn, radial

    if upper:
        nc = LASH_CLUMPS_UP
        # clump centres: denser toward the outer half (the fringe thickens outward), small jitter
        uc = (np.arange(nc) + 0.5 + rng.uniform(-0.25, 0.25, nc)) / nc
        uc = 0.04 + 0.94 * uc
        du = 0.92 / nc
        ks = rng.integers(LASH_PER_CLUMP[0], LASH_PER_CLUMP[1] + 1, nc)
        ks = np.where(uc < 0.2, np.minimum(ks, 4), ks)                        # thin at the inner corner
        cid = np.repeat(np.arange(nc), ks)
        u = np.clip(uc[cid] + rng.uniform(-0.5, 0.5, len(cid)) * du * 0.95, 0.03, 0.985)
        roots, tg, nn, radial = at(u)
        _, _, nnc, radc = at(uc)
        # clump direction: out of the rim (radial + lid normal) then up; outer third sweeps laterally (the wing)
        sweep = 0.55 * smoothstep(0.55, 1.0, uc) - 0.18 * smoothstep(0.22, 0.0, uc)
        dc = nrm(0.6 * radc + 0.55 * nnc + 0.08 * vert + lat * sweep[:, None] + rng.normal(0, 0.015, (nc, 3)))
        Lcl = er * (0.09 + 0.11 * smoothstep(0.25, 0.85, uc) - 0.05 * smoothstep(0.93, 1.0, uc)) * rng.uniform(0.9, 1.08, nc)
        curl_c = 1.2 + 0.25 * smoothstep(0.6, 1.0, uc)
        curl_dir = nrm(vert[None] - 0.4 * nnc)
        Xc, t = _lash_shape(at(uc)[0], dc, curl_dir, Lcl, curl_c, ec, er)   # clump leaders (centre strands)
        # member strands: own root, length 0.8-1.0 of the leader, tips converge to the leader's tip (wet clump),
        # roots fan slightly so the clump is a tapered wedge
        d = nrm(dc[cid] + rng.normal(0, 0.04, (len(cid), 3)))
        Lm = Lcl[cid] * rng.uniform(0.78, 1.0, len(cid))
        Xm, _ = _lash_shape(roots, d, curl_dir[cid], Lm, curl_c[cid], ec, er)
        conv = (rng.uniform(0.85, 0.97, len(cid))[:, None] * t[None] ** 1.4)[..., None]
        lead = Xc[cid] - Xc[cid, :1] + roots[:, None]                          # the leader's shape from this root
        X = Xm + conv * (lead - Xm)
        # tip convergence: last points pull toward the leader's actual tip
        tipw = (np.clip((t - 0.55) / 0.45, 0, 1) ** 2 * 0.6)[None, :, None] * (Lm / Lcl[cid])[:, None, None]
        X = X + tipw * (Xc[cid] - X)
    else:
        n = LASH_N_LO
        u = np.sort(0.22 + 0.74 * rng.random(n) ** 0.8)                        # sparse, mostly the outer half
        roots, tg, nn, radial = at(u)
        d0 = nrm(0.55 * radial + 0.5 * nn + 0.3 * vert + lat[None] * (0.35 * smoothstep(0.6, 1.0, u))[:, None]
                 + rng.normal(0, 0.05, (n, 3)))
        L = er * (0.035 + 0.04 * smoothstep(0.4, 0.95, u)) * rng.uniform(0.7, 1.1, n)
        curl_dir = nrm(np.tile(vert, (n, 1)) - 0.4 * nn)
        X, t = _lash_shape(roots, d0, curl_dir, L, 0.35, ec, er)
        cid = np.arange(n)
    for k in range(1, P_LASH):
        dd = X[:, k] - ec
        r = np.linalg.norm(dd, axis=1)
        bad = r < er * 1.1
        X[bad, k] = ec + dd[bad] / r[bad, None] * er * 1.1
    X[:, 2:] = hs.push(X[:, 2:], 0.0003)
    X[:, 0] = roots
    return X, cid


# ------------------------------------------------------------------------------------------------ assembly
def _subset(rng, S, n, strat=None):
    if n >= S:
        return np.arange(S)
    if strat is None:
        return np.sort(rng.choice(S, n, replace=False))
    return np.sort(rng.choice(S, n, replace=False))


def _arrays(X, wr, wt, clump, rng, wscale=1.0):
    S = len(X)
    jit = rng.uniform(0.8, 1.2, S)
    return {"pos": X.astype(np.float32),
            "width": np.stack([wr * jit * wscale, wt * jit * wscale], 1).astype(np.float32),
            "seed": rng.random(S).astype(np.float32), "clump": np.asarray(clump, np.uint32)}


def build(base, S, scale=1.0):
    rng = np.random.default_rng(RNG_SEED)
    hs = HeadSurf(base, rng=rng)
    out = {}
    nsc = int(N_LOD0["scalp"] * scale)
    if STYLE == "bob":
        import hair_bob
        SX, sclump, slayer, fl = hair_bob.bob_groom(base, hs, rng, nsc)
    else:
        bun = Bun(hs, base)
        SX, sclump, slayer = scalp_groom(base, hs, bun, rng, nsc)
    if TENDRILS:   # ASSET2 v5: face-framing locks ride in the scalp group (OFF: wiry + a hard shadow line on the cheek, h2/h3)
        TX, tcl = tendrils(base, hs, bun, rng)
        tcl = tcl - 900000 + int(sclump.max()) + 1
        SX = np.concatenate([SX, TX]); sclump = np.concatenate([sclump, tcl]); slayer = np.concatenate([slayer, np.ones(len(TX))])
    X0, c0 = SX, sclump
    wr, wt = 8e-5, 2.5e-5
    hair_meta = {"bone": "head", "follow": "bone", "surface_submesh": "head", "width_root_m": wr, "width_tip_m": wt,
                 "color_root": HAIR_ROOT, "color_tip": HAIR_TIP, "melanin": 0.9, "redness": 0.5,
                 "roughness": 0.28, "radial_roughness": 0.35, "coat": 0.2,
                 "clumps": int(c0.max()) + 1,
                 "groom": "side-parted wavy bob, left side tucked" if STYLE == "bob" else "slick back, low bun"}
    si = _subset(rng, len(SX), N_LOD1["scalp"])
    w1 = np.sqrt(len(SX) / len(si)) * 0.75
    out["scalp"] = {"meta": hair_meta, "lod0": _arrays(X0, wr, wt, c0, rng),
                    "lod1": _arrays(SX[si], wr, wt, sclump[si], rng, w1)}
    if STYLE == "bob":
        FX = hair_bob.bob_fuzz(base, hs, fl, rng, SX, slayer, int(N_LOD0["fuzz"] * scale))
    else:
        FX = fuzz(base, hs, bun, rng, SX, slayer, int(N_LOD0["fuzz"] * scale))
    fi = _subset(rng, len(FX), N_LOD1["fuzz"])
    fmeta = dict(hair_meta, width_root_m=4.5e-5, width_tip_m=1.2e-5, roughness=0.38)
    out["fuzz"] = {"meta": fmeta, "lod0": _arrays(FX, 4.5e-5, 1.2e-5, 200000 + np.arange(len(FX)), rng),
                   "lod1": _arrays(FX[fi], 4.5e-5, 1.2e-5, 200000 + fi, rng, 1.3)}
    BW = (1.25e-4, 3.5e-5)   # ASSET2 v5: finer strands, more of them (judge: brows read as sausage strokes)
    brow_meta = {"bone": "head", "follow": "surface", "surface_submesh": "head", "width_root_m": BW[0],
                 "width_tip_m": BW[1], "color_root": [0.050, 0.028, 0.016], "color_tip": [0.11, 0.065, 0.038],
                 "melanin": 0.85, "redness": 0.5, "roughness": 0.45}
    for side in ("l", "r"):
        BX = brows(base, hs, rng, side, N_LOD0["brow"])
        bi = _subset(rng, len(BX), N_LOD1["brow"])
        out[f"brows_{side}"] = {"meta": brow_meta, "lod0": _arrays(BX, BW[0], BW[1], np.arange(len(BX)), rng),
                                "lod1": _arrays(BX[bi], BW[0], BW[1], bi, rng, np.sqrt(len(BX) / len(bi)) * 0.8)}
    out.update(lash_groups(base, hs))
    return out


LASH_W_UP = (2.6e-4, 3e-5)      # asset-eyes4: root / tip width (m), upper clumps
LASH_W_LO = (0.9e-4, 2e-5)


def lash_groups(base, hs):
    """asset-eyes4: the four lash groups (brown, clumped upper fringe, sparse lower), own rng (lash changes never
    reshuffle the brows), the same draw for both sides (a mirrored fringe). lod1 = lod0 (few strands)."""
    out = {}
    lash_meta = {"bone": "head", "follow": "surface", "surface_submesh": "head", "width_root_m": LASH_W_UP[0],
                 "width_tip_m": LASH_W_UP[1], "color_root": LASH_COL_ROOT, "color_tip": LASH_COL_TIP,
                 "melanin": 0.95, "redness": 0.45, "roughness": 0.32, "groom": "eyes4 clumped fringe"}
    for side in ("l", "r"):
        lrng = np.random.default_rng(RNG_SEED + 17)
        for nm, up in (("upper", True), ("lower", False)):
            AX, acl = lashes(base, hs, lrng, side, up, 0)
            wr_, wt_ = LASH_W_UP if up else LASH_W_LO
            m_ = dict(lash_meta, width_root_m=wr_, width_tip_m=wt_)
            if not up:
                m_.update(color_root=[0.045, 0.028, 0.019], color_tip=[0.08, 0.052, 0.036])
            a0 = _arrays(AX, wr_, wt_, acl, lrng)
            out[f"lashes_{nm}_{side}"] = {"meta": m_, "lod0": a0, "lod1": a0}
    return out
