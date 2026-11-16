"""HAIR-BL — realistic eyelashes for a BUILT head (junor.asset vN), derived from the asset alone.

    build(H, rng, log) -> {"lashes_upper_l", "lashes_lower_l", "lashes_upper_r", "lashes_lower_r"}

Lid margin: the head skin's own edge loop through the lid_upper_* / lid_lower_* landmarks (they are head vertices on
the ICT margin band): a shortest edge path (welded head graph) through the landmark vertices, smoothed and resampled
inner -> outer. Roots sit on the ANTERIOR margin (offset 0.15-0.5 mm forward along the skin, 2-3 irregular rows) and
are exact closest points on the head triangles, so surface follow binds them to the margin triangles and they ride
(and rotate with) the blink. Upper: ~135 per eye, 6-11 mm (longest at the outer third), emerging forward and a little
down, curling up 60-80 deg, the outer third sweeping laterally, gentle clumps of 2-4 whose tips converge, a few
crossing. Lower: ~60 per eye, 3-6 mm, sparse at the inner third, curling down. No point closer to the eyeball centre
than the cornea apex (+0.4 mm).
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import dijkstra

try:
    from head import nrm, smoothstep, submesh_tris
    from brows import MeshProj, arc_resample
except ImportError:
    from .head import nrm, smoothstep, submesh_tris
    from .brows import MeshProj, arc_resample

P_LASH = 12
N_UP, N_LO = 66, 22          # HAIR2 r8: fewer (lead: false-lash look)
W_UP = (1.4e-4, 2.0e-5)     # root / tip width (m); HAIR2 r8 finer (was 2.0e-4)
W_LO = (0.5e-4, 0.6e-5)
COL_ROOT = [0.009, 0.0065, 0.005]
COL_TIP = [0.022, 0.015, 0.011]
COL_LO_ROOT = [0.030, 0.020, 0.014]     # lower lashes: much lighter / finer (r4)
COL_LO_TIP = [0.060, 0.042, 0.030]
# length profiles (mm) along the margin, u = 0 inner corner .. 1 outer corner
UP_U = np.array([0.10, 0.25, 0.45, 0.65, 0.82, 0.93, 0.99])
UP_L = np.array([7.0, 8.8, 10.3, 11.4, 12.0, 11.0, 8.6]) * 0.62   # HAIR2 r8: -38 % (lashes reached the brow shadow)
LO_U = np.array([0.36, 0.50, 0.65, 0.80, 0.92, 0.98])
LO_L = np.array([2.6, 3.2, 3.9, 4.4, 4.0, 3.0])


class Margin:
    """the welded head-edge graph, shortest paths through lid landmarks"""

    def __init__(self, H):
        V = H.V
        T, _ = submesh_tris(H.A)
        key = np.round(V / 1e-6).astype(np.int64)
        _, inv = np.unique(key, axis=0, return_inverse=True)
        inv = inv.ravel()
        TT = inv[T]
        E = np.unique(np.sort(np.concatenate([TT[:, [0, 1]], TT[:, [1, 2]], TT[:, [2, 0]]]), 1), axis=0)
        n = inv.max() + 1
        rep = np.zeros(n, np.int64)
        rep[inv] = np.arange(len(inv))
        self.W = V[rep]                                        # welded positions
        w = np.linalg.norm(self.W[E[:, 0]] - self.W[E[:, 1]], axis=1)
        self.G = sp.coo_matrix((w, (E[:, 0], E[:, 1])), shape=(n, n)).tocsr()
        used = np.unique(TT)
        from scipy.spatial import cKDTree
        self.kd = cKDTree(self.W[used])
        self.used = used

    def snap(self, P):
        return self.used[self.kd.query(P)[1]]

    def path(self, pts):
        ids = self.snap(np.asarray(pts, float))
        out = [ids[0]]
        for a, b in zip(ids[:-1], ids[1:]):
            _, pred = dijkstra(self.G, directed=False, indices=a, return_predecessors=True, limit=0.03)
            seq = []
            c = b
            while c != a and c >= 0:
                seq.append(c)
                c = pred[c]
            out += seq[::-1]
        return self.W[np.array(out)]


def _margin_curve(H, MP, MG, side, upper, n=240):
    lm = H.A.meta["landmarks"]
    sg = 1.0 if side == "l" else -1.0
    P = np.asarray(lm[f"lid_{'upper' if upper else 'lower'}_{side}"], float)[:, :3]
    ec = np.asarray(lm[f"eye_{side}"], float)
    if sg * (P[0, 0] - P[-1, 0]) > 0:                         # inner (medial) -> outer
        P = P[::-1]
    C = MG.path(P)
    C, L = arc_resample(C, n)
    for _ in range(6):
        C[1:-1] = 0.25 * C[:-2] + 0.5 * C[1:-1] + 0.25 * C[2:]
    C, N, _ = MP.project(C)
    return C, N, L, ec, sg


def _shape(roots, d0, cdir, L, curl, pw=0.95):
    """march P_LASH points: direction rotates from d0 toward cdir by curl * t^pw"""
    n = len(roots)
    t = np.linspace(0, 1, P_LASH)
    X = np.zeros((n, P_LASH, 3))
    X[:, 0] = roots
    p = roots.copy()
    ds = (L / (P_LASH - 1))[:, None]
    cd = nrm(cdir - d0 * (cdir * d0).sum(1, keepdims=True))
    for k in range(1, P_LASH):
        a = (curl * (0.5 * (t[k] + t[k - 1])) ** pw)[:, None]
        dk = nrm(d0 * np.cos(a) + cd * np.sin(a))
        p = p + dk * ds
        X[:, k] = p
    return X, t


def lash_side(H, MP, MG, rng, side, upper, n_target, log=print, R_keep=None, blink=None):
    C, Nc, Ltot, ec, sg = _margin_curve(H, MP, MG, side, upper)
    m = len(C)
    fwd = np.array([0.0, 0.0, 1.0])
    lat = np.array([sg, 0.0, 0.0])
    vert = np.array([0.0, 1.0 if upper else -1.0, 0.0])
    # roots: stratified along the margin (irregular spacing), slight density bias to the outer half (upper)
    if upper:
        u = (np.arange(n_target) + rng.uniform(-0.35, 1.35, n_target)) / n_target   # irregular spacing
        u = 0.08 + 0.91 * np.clip(u, 0.0, 1.0) ** 0.93
        Lp = np.interp(u, UP_U, UP_L)
    else:
        u = (np.arange(n_target) + rng.uniform(0.05, 0.95, n_target)) / n_target
        u = 0.36 + 0.60 * np.clip(u, 0.0, 1.0) ** 0.85                                  # none in the inner third
        Lp = np.interp(u, LO_U, LO_L)
    i = np.clip(np.rint(u * (m - 1)).astype(int), 0, m - 1)
    p0, n0 = C[i], Nc[i]
    tg = nrm(np.gradient(C, axis=0))[i]
    # roots on the ANTERIOR lid skin just past the margin band: march over the band (its triangles face down for the
    # upper lid and, in the ICT blink, roll the wrong way = lashes would flip up on a closed eye) until the skin faces
    # forward, then 0-0.3 mm further (2-3 irregular rows). The pretarsal skin rotates WITH the lid on a blink.
    q = p0 + tg * rng.normal(0, 0.06e-3, len(u))[:, None]
    # across the band toward the front: perpendicular to the margin tangent and to the band's own normal
    q, n_band, _ = MP.project(q)
    acr = nrm(np.cross(tg, n_band))
    acr[acr[:, 2] < 0] *= -1
    done = np.zeros(len(u), bool)

    def march_dir(nq):
        return nrm(acr - nq * (nq * acr).sum(1, keepdims=True))

    def d_el(t):
        """elevation change (deg) of a head triangle's normal at a full blink (+ = the frame rolls UP)"""
        TT = MP.T[t]
        P0 = MP.V[TT]
        P1 = P0 + blink[TT]
        f = lambda P: nrm(np.cross(P[:, 1] - P[:, 0], P[:, 2] - P[:, 0]))[:, 1]
        return np.degrees(np.arcsin(np.clip(f(P1), -1, 1)) - np.arcsin(np.clip(f(P0), -1, 1)))
    # stop where the skin turns forward (band normal rotated > 50 deg) AND - when the asset has the blink - where the
    # root triangle rolls DOWN with the closing lid (the ICT margin band rolls up ~40-80 deg: upper lashes would stand
    # up on a closed eye); <= 1.8 mm
    for _ in range(23):
        q, nq, _ = MP.project(q)
        ok = (nq * n_band).sum(1) < np.cos(np.radians(50))
        if blink is not None:
            ok = d_el(MP.last_tri) < 10.0 if upper else ok | (np.abs(d_el(MP.last_tri)) < 8.0)
        done |= ok
        if done.all():
            break
        q = np.where(done[:, None], q, q + march_dir(nq) * 0.08e-3)
    extra = rng.choice([0.0, 0.12, 0.24], len(u), p=[0.45, 0.35, 0.2]) * 1e-3 + np.abs(rng.normal(0, 0.03e-3, len(u)))
    q, nq, _ = MP.project(q)
    roots, nn, _ = MP.project(q + march_dir(nq) * extra[:, None])
    self_off = np.linalg.norm(roots - p0, axis=1)
    n = len(roots)
    # out-of-aperture direction (image plane), from the eye centre
    eo = roots - ec
    eo[:, 2] = 0
    eo = nrm(eo + vert * 0.004)
    eo = nrm(vert * 0.7 + eo * 0.3)        # mostly up / down: a purely radial curl flares the corner lashes sideways
    # sweep: inner lashes lean to the nose a little, outer third sweeps out (the wing)
    # r4: the visible lash starts AT the lid margin: the root (blink-safe, on the lid skin) is joined by one short
    # skin-hugging segment to an anchor on the margin's front edge, and the curved lash grows from there (open-eyed,
    # nothing dark sits on the lid skin above the lash line; on a blink the root frame still carries the lash down)
    anchor = p0 + (roots - p0) * 0.3
    anchor, na, _ = MP.project(anchor)
    anchor = anchor + na * 0.12e-3
    if upper:
        sweep = -0.35 * (1 - smoothstep(0.05, 0.4, u)) + 0.8 * smoothstep(0.45, 1.0, u)   # a fan, like the concept
        d0 = nrm(fwd * 1.0 - eo * 0.05 + lat * sweep[:, None])     # out of the margin, forward and a touch up
        curl = np.radians(rng.uniform(55, 78, n) + 8 * smoothstep(0.5, 0.95, u))
        cdir = nrm(eo + 0.1 * fwd)
    else:
        sweep = -0.1 * (1 - smoothstep(0.2, 0.4, u)) + 0.2 * smoothstep(0.6, 1.0, u)
        d0 = nrm(fwd * 0.75 + eo * 0.65 + lat * sweep[:, None])
        curl = np.radians(rng.uniform(18, 34, n))
        cdir = nrm(eo)
    cross = rng.random(n) < (0.07 if upper else 0.05)
    d0 = nrm(d0 + lat * (np.where(cross, rng.choice([-1, 1], n) * rng.uniform(0.25, 0.5, n), 0.0)
                         + rng.normal(0, 0.06, n))[:, None] + rng.normal(0, 0.035, (n, 3)))
    L = Lp * 1e-3 * rng.uniform(0.82, 1.1, n)
    short = rng.random(n) < 0.12                                      # new / broken short lashes
    L = np.where(short, L * rng.uniform(0.4, 0.7, n), L)
    Xc, _ = _shape(anchor, d0, cdir, L, curl)
    X = np.concatenate([roots[:, None], Xc], 1)
    NP = X.shape[1]
    t = np.linspace(0, 1, NP)
    # gentle clumps: runs of 2-4 neighbours along the margin, tips pulled toward the clump's mean shape
    cid = np.zeros(n, np.int64)
    c, k = 0, 0
    while k < n:
        sz = int(rng.choice([1, 2, 3, 4], p=[0.35, 0.4, 0.25, 0.0] if upper else [0.85, 0.15, 0.0, 0.0]))
        cid[k:k + sz] = c
        c += 1
        k += sz
    for cc in range(c):
        mm = np.where(cid == cc)[0]
        if len(mm) < 2:
            continue
        rel = X[mm] - X[mm, :1]
        mean = rel.mean(0)
        pull = rng.uniform(0.3, 0.6) * smoothstep(0.45, 1.0, t)[None, :, None]
        X[mm] = X[mm, :1] + rel + pull * (mean[None] - rel)
    # keep-out: never closer to the eyeball centre than the cornea apex (+ margin)
    if R_keep is not None:
        for k in range(1, NP):
            dd = X[:, k] - ec
            r = np.linalg.norm(dd, axis=1)
            rmin = np.maximum(R_keep, np.linalg.norm(roots - ec, axis=1) * 0.999)
            bad = r < rmin
            X[bad, k] = ec + dd[bad] / r[bad, None] * rmin[bad, None]
    # keep-off the lid skin itself (curling back onto the lid): >= 0.25 mm above the head skin beyond the root
    S_, Ns_, dist = MP.project(X[:, 3:])
    sdv = ((X[:, 3:] - S_) * Ns_).sum(2)
    bad = sdv < 0.25e-3
    X[:, 3:][bad] += (0.25e-3 - sdv[bad])[:, None] * Ns_[bad]
    X[:, 0] = roots
    log(f"[lashes] {side} {'upper' if upper else 'lower'}: {n} strands, margin {Ltot * 1000:.1f} mm, "
        f"L {L.min() * 1e3:.1f}-{L.max() * 1e3:.1f} mm, clumps {c}, root off margin {np.median(self_off) * 1e3:.2f} mm")
    return X, cid


def build(H, rng, log=print):
    MP = MeshProj(H)
    MG = Margin(H)
    out = {}
    lrng0 = np.random.default_rng(int(rng.integers(1 << 30)))
    seeds = lrng0.integers(1 << 30, size=4)
    for si, side in enumerate(("l", "r")):
        # keep-out radius: the cornea apex distance from the eye centre (+0.4 mm)
        ec = np.asarray(H.A.meta["landmarks"][f"eye_{side}"], float)
        R_keep = None
        for s in H.A.submeshes:
            if s.get("name") == f"cornea_{side}":
                Vc = H.V[s["first_vertex"]:s["first_vertex"] + s["vertex_count"]]
                R_keep = float(np.linalg.norm(Vc - ec, axis=1).max()) + 0.4e-3
        blink = None
        nmB = "eyeBlinkLeft" if side == "l" else "eyeBlinkRight"
        if nmB in H.A.shape_names:
            blink = np.zeros_like(H.V)
            for snm, ii, dp, _ in H.A.shapes:
                if snm in (nmB, f"corr_{nmB}_{nmB}"):
                    blink[ii] += dp
        for ui, (nm, up) in enumerate((("upper", True), ("lower", False))):
            lrng = np.random.default_rng(seeds[2 * ui + 0] + (7 if side == "r" else 0))
            X, cid = lash_side(H, MP, MG, lrng, side, up, N_UP if up else N_LO, log, R_keep, blink)
            W = W_UP if up else W_LO
            S = len(X)
            ws = lrng.uniform(0.8, 1.15, S)
            arr = {"pos": X.astype(np.float32),
                   "width": np.stack([W[0] * ws, W[1] * ws], 1).astype(np.float32),
                   "seed": lrng.random(S).astype(np.float32),
                   "clump": (cid + (400000 if up else 410000) + (5000 if side == "r" else 0)).astype(np.uint32)}
            meta = {"bone": "head", "follow": "surface", "surface_submesh": "head", "width_root_m": W[0],
                    "width_tip_m": W[1], "color_root": COL_ROOT if up else COL_LO_ROOT, "color_tip": COL_TIP if up else COL_LO_TIP, "melanin": 0.95,
                    "redness": 0.4, "roughness": 0.32, "groom": f" {nm} lashes (margin roots, clumps 2-4)"}
            out[f"lashes_{nm}_{side}"] = {"meta": meta, "lod0": arr, "lod1": arr}
    return out
