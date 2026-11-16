"""junor — wardrobe neckline helpers (realistic adult refit of the juno wardrobe).

The juno wardrobe only knew the MH body; on junor the MH faces above the ICT neck loop (base.cut_faces) are never rendered,
the ICT scan skin is there instead. So:

  TrueSkin(base)            triangle soup of what is really rendered: MH body minus cut faces + ICT face/head-neck skin +
                            the weld bridge. ray hits, closest points, dense outward samples.
  proxy_positions(base, sk) MH rest positions with every vertex that only belongs to cut faces moved onto the ICT skin
                            (closest point): garment shells derived from MH faces + the BodySDF then follow the ICT neck.
  NeckFrame(base, sk)       explicit neckline curves: the blouse collar base (front at the jugular notch, high at the side neck
                            points, back at C7) and the cardigan neckline (same saddle a touch lower/wider + the V).
  Layers                    outward-oriented dense samples of finished garment surfaces; push(X, clr) lifts points so they
                            lie at least clr OUTSIDE every layer they sit over (fixes the white blouse bits poking through
                            the cardigan: the collar is pushed over shell AND rib band, not just the shell).
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from juno_base import tri_of, vertex_normals
from wardrobe_geo import normalize, smoothstep

NECK_Y_FRONT = 0.018    # collar base above the weld loop's front point (v1: 0.025)
NECK_Y_BACK = 0.030     # above the loop's back point (v1: 0.034)
NECK_Y_SIDE = 0.068     # side neck point above the loop's front point (v1: 0.083)

BARY = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1 / 3, 1 / 3, 1 / 3], [.5, .5, 0], [0, .5, .5], [.5, 0, .5],
                 [2 / 3, 1 / 6, 1 / 6], [1 / 6, 2 / 3, 1 / 6], [1 / 6, 1 / 6, 2 / 3]])


def dense_samples(P, T, N=None):
    """points + interpolated normals at 10 barycentric spots per triangle."""
    if N is None:
        N = vertex_normals(P, T)
    X = np.einsum("kj,tjc->tkc", BARY, P[T]).reshape(-1, 3)
    Nx = normalize(np.einsum("kj,tjc->tkc", BARY, N[T]).reshape(-1, 3))
    return X, Nx


def closest_on_tris(P, tris, Q, k=16):
    cen = P[tris].mean(1)
    _, cand = cKDTree(cen).query(Q, k=min(k, len(tris)))
    best_t = np.zeros(len(Q), np.int64)
    best_b = np.zeros((len(Q), 3))
    best_d = np.full(len(Q), np.inf)
    for j in range(cand.shape[1]):
        t = cand[:, j]
        a, b, c = P[tris[t, 0]], P[tris[t, 1]], P[tris[t, 2]]
        v0, v1, v2 = b - a, c - a, Q - a
        d00, d01, d11 = (v0 * v0).sum(1), (v0 * v1).sum(1), (v1 * v1).sum(1)
        d20, d21 = (v2 * v0).sum(1), (v2 * v1).sum(1)
        den = np.maximum(d00 * d11 - d01 * d01, 1e-18)
        v = (d11 * d20 - d01 * d21) / den
        w = (d00 * d21 - d01 * d20) / den
        bc = np.clip(np.stack([1 - v - w, v, w], 1), 0, 1)
        bc /= bc.sum(1, keepdims=True)
        q = bc[:, 0:1] * a + bc[:, 1:2] * b + bc[:, 2:3] * c
        d = np.linalg.norm(q - Q, axis=1)
        m = d < best_d
        best_d[m], best_t[m], best_b[m] = d[m], t[m], bc[m]
    return best_t, best_b, best_d


class TrueSkin:
    def __init__(self, base):
        H = base.ict
        body = [fi for fi in base.faces_of("body") if fi not in base.cut_faces]
        T1 = np.array([t for fi in body for t in tri_of([v for v, _ in base.obj.faces[fi]])], np.int64)
        F, _ = H.faces("face", "headneck")
        T2 = np.concatenate([F[:, [0, 1, 2]], F[F[:, 3] >= 0][:, [0, 2, 3]]])
        br = base.bridge
        n1, n2 = len(base.VA), len(H.P)
        self.P = np.concatenate([base.VA, H.P, br["P"]])
        self.T = np.concatenate([T1, T2 + n1, br["tris"] + n1 + n2])
        self.ict_T = T2 + n1
        self.N = vertex_normals(self.P, self.T)
        self.S, self.SN = dense_samples(self.P, self.T, self.N)
        self.tree = cKDTree(self.S)
        self.n_mh = len(T1)

    def visible_only(self):
        """the ICT skin + bridge alone (the MH skin under the cloth is hidden; juno's cloth is built on a flattened chest
        proxy, so pushing cloth out of the real MH bust would make the blouse poke through the knit)."""
        o = TrueSkin.__new__(TrueSkin)
        o.P, o.T, o.N, o.ict_T = self.P, self.T[self.n_mh:], self.N, self.ict_T
        o.S, o.SN = dense_samples(o.P, o.T, o.N)
        o.tree = cKDTree(o.S)
        o.n_mh = 0
        return o

    def query(self, X):
        _, i = self.tree.query(X)
        q, n = self.S[i], self.SN[i]
        return ((X - q) * n).sum(1), n, q

    def push_out(self, X, clr, iters=3, reach=0.03):
        """lift points that are inside / closer than clr to the skin; only near the skin (reach), so a far point whose
        nearest sample is an unrelated patch (e.g. the neck seen from the lower back) is never moved."""
        X = np.array(X, float)
        clr = np.broadcast_to(np.asarray(clr, float), (len(X),))
        for _ in range(iters):
            s, n, q = self.query(X)
            v = X - q
            lat = np.linalg.norm(v - s[:, None] * n, axis=1)
            bad = (s < clr) & (s > -reach) & (lat < 0.01)
            if not bad.any():
                break
            X[bad] += n[bad] * (clr[bad] - s[bad])[:, None]
        return X

    def ray(self, O, D, tris=None):
        """first hit (t>0) of rays O + t D on the skin -> (points, normals, ok)."""
        T = self.T if tris is None else tris
        a, b, c = self.P[T[:, 0]], self.P[T[:, 1]], self.P[T[:, 2]]
        e1, e2 = b - a, c - a
        out_p = np.zeros((len(O), 3))
        out_n = np.zeros((len(O), 3))
        ok = np.zeros(len(O), bool)
        for k in range(len(O)):
            o, d = O[k], D[k]
            p = np.cross(d, e2)
            det = (e1 * p).sum(1)
            good = np.abs(det) > 1e-14
            inv = np.where(good, 1.0 / np.where(good, det, 1), 0)
            s = o - a
            u = (s * p).sum(1) * inv
            q = np.cross(s, e1)
            v = (q @ d) * inv
            t = (e2 * q).sum(1) * inv
            hit = good & (u >= 0) & (v >= 0) & (u + v <= 1) & (t > 1e-6)
            if hit.any():
                j = np.where(hit)[0][np.argmin(t[hit])]
                out_p[k] = o + d * t[j]
                nn = self.N[T[j]].mean(0)
                out_n[k] = nn / np.linalg.norm(nn)
                ok[k] = True
        return out_p, out_n, ok


def proxy_positions(base, sk):
    """MH rest positions with the vertices of cut faces (only) moved onto the ICT skin."""
    V = base.VA.copy()
    keep = set()
    cutv = set()
    for fi in base.faces_of("body"):
        vs = [v for v, _ in base.obj.faces[fi]]
        (cutv if fi in base.cut_faces else keep).update(vs)
    only_cut = np.array(sorted(cutv - keep), np.int64)
    t, b, _ = closest_on_tris(sk.P, sk.ict_T, V[only_cut])
    V[only_cut] = (sk.P[sk.ict_T[t]] * b[:, :, None]).sum(1)
    return V


class NeckFrame:
    """explicit neckline curves on the true skin (angle th = atan2(x, z - zc): 0 front, +-pi back, + = her left)."""

    def __init__(self, base, sk):
        H = base.ict
        L = H.P[H.neck_loop]
        th = np.arctan2(L[:, 0], L[:, 2] - L[:, 2].mean())
        yf = L[np.argmin(np.abs(th)), 1]
        yb = L[np.argmin(np.abs(np.abs(th) - np.pi)), 1]
        # collar base: jugular notch (front), side neck point (highest), C7 (back)
        # round 3 (lead 11-16, v2 neck 1.2 cm shorter): lower collar base so the neck reads long in the bust framing
        self.y_front = yf + NECK_Y_FRONT
        self.y_back = yb + NECK_Y_BACK
        self.y_side = yf + NECK_Y_SIDE
        self.zc = float(base.bones["neck_01"]["head"][2] + 0.02)
        self.sk = sk

    def y_of(self, th, drop=0.0):
        a = 0.5 * (0.5 * (self.y_front + self.y_back) + self.y_side)
        c = 0.5 * (self.y_front + self.y_back) - a
        b = 0.5 * (self.y_front - self.y_back)
        return a + b * np.cos(th) + c * np.cos(2 * th) - drop

    def theta(self, X):
        return np.arctan2(X[:, 0], X[:, 2] - self.zc)

    def above(self, X, drop=0.0):
        """True where X lies above the neckline saddle (only meaningful near the neck/shoulders)."""
        return X[:, 1] > self.y_of(self.theta(X), drop)

    def curve(self, ths, drop=0.0):
        """skin points + normals of the saddle curve at the angles ths (horizontal rays from the neck axis)."""
        y = self.y_of(ths, drop)
        O = np.stack([np.zeros_like(ths), y, np.full_like(ths, self.zc)], 1)
        D = np.stack([np.sin(ths), np.zeros_like(ths), np.cos(ths)], 1)
        near = (np.abs(self.sk.P[self.sk.T].mean(1)[:, 1] - y.mean()) < 0.12)
        P, N, ok = self.sk.ray(O, D, self.sk.T[near])
        if not ok.all():
            raise RuntimeError("neckline ray missed the skin")
        return P, N


class Layers:
    """outward-oriented dense samples of garment / skin surfaces (several layers), for lay-over push-outs."""

    def __init__(self, sk):
        self.sk = sk
        self.L = []

    def add(self, P, T, lat=None):
        P = np.asarray(P, float)
        T = np.asarray(T, np.int64)
        X, N = dense_samples(P, T)
        _, n_sk, _ = self.sk.query(X)
        dt = (N * n_sk).sum(1)
        N = N * np.where(dt < 0, -1.0, 1.0)[:, None]
        keep = np.abs(dt) > 0.3                                # drop side walls of bands
        X, N = X[keep], N[keep]
        if lat is None:
            e = np.linalg.norm(P[T[:, 0]] - P[T[:, 1]], axis=1)
            lat = float(np.median(e)) * 0.6 + 0.002
        self.L.append((cKDTree(X), X, N, lat))

    def push(self, X, clr, iters=4, depth=0.03):
        X = np.array(X, float)
        clr = np.broadcast_to(np.asarray(clr, float), (len(X),))
        for _ in range(iters):
            best = np.zeros(len(X))
            bn = np.zeros((len(X), 3))
            for tree, S, N, lat in self.L:
                _, idx = tree.query(X, k=8)
                q, n = S[idx], N[idx]
                v = X[:, None, :] - q
                d = (v * n).sum(2)
                lt = np.linalg.norm(v - d[..., None] * n, axis=2)
                need = np.where((lt < lat) & (d > -depth), clr[:, None] - d, 0.0)
                j = np.argmax(need, 1)
                nb = need[np.arange(len(X)), j]
                m = nb > best
                best[m] = nb[m]
                bn[m] = n[np.arange(len(X)), j][m]
            if best.max() < 1e-5:
                break
            X += bn * best[:, None]
        return X


def smoothstep_(a, b, x):
    return smoothstep(a, b, x)
