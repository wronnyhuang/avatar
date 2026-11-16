"""asset — wardrobe geometry helpers (owner: wardrobe helper).

Garments are derived from the MH body surface of whatever Base the lead builds (procedural, landmark-driven):
  BodySDF        nearest-surface signed distance to the skin (kd-tree over dense samples + normals)
  QuadMesh       welded quad mesh with per-vertex attributes (body anchor position, dense skin weights)
  subdivide      linear quad split (attributes interpolated) — then smoothing/collision rounds it
  sweep          a rounded-rectangle profile swept along a curve (rib bands, cuffs, hems, soles)
  to_part        per-corner UVs -> split at UV seams (weld ids keep normals continuous) -> parts.Part
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from parts import Part


def normalize(v):
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-12)


def smoothstep(a, b, x):
    t = np.clip((x - a) / (b - a), 0, 1)
    return t * t * (3 - 2 * t)


def quad_normals(P, Q):
    T = np.concatenate([Q[:, [0, 1, 2]], Q[:, [0, 2, 3]]])
    fn = np.cross(P[T[:, 1]] - P[T[:, 0]], P[T[:, 2]] - P[T[:, 0]])
    n = np.zeros_like(P)
    for k in range(3):
        np.add.at(n, T[:, k], fn)
    return normalize(n)


def tri_normals(P, T):
    fn = np.cross(P[T[:, 1]] - P[T[:, 0]], P[T[:, 2]] - P[T[:, 0]])
    n = np.zeros_like(P)
    for k in range(3):
        np.add.at(n, T[:, k], fn)
    return normalize(n)


class BodySDF:
    """Approximate signed distance to the skin: d = dot(p - q, n_q), q = nearest dense sample of the body surface."""

    def __init__(self, base, exclude_bones=()):
        fids = base.faces_of("body")
        Q = np.array([[vi for vi, _ in base.obj.faces[f]] for f in fids], np.int64)
        V, N = base.VA, base.N
        pts = [V[np.unique(Q)], ]
        nrm = [N[np.unique(Q)]]
        # face centres + edge midpoints + quarter points for density (~5 mm spacing on the torso)
        for wts in ([.25, .25, .25, .25], [.5, .5, 0, 0], [0, .5, .5, 0], [.375, .375, .125, .125], [.125, .375, .375, .125],
                    [.125, .125, .375, .375], [.375, .125, .125, .375]):
            w = np.array(wts)[None, :, None]
            pts.append((V[Q] * w).sum(1))
            nrm.append(normalize((N[Q] * w).sum(1)))
        self.P = np.concatenate(pts)
        self.Nn = np.concatenate(nrm)
        self.tree = cKDTree(self.P)

    def query(self, X, k=1):
        d, i = self.tree.query(X, k=k)
        if k > 1:
            i = i[:, 0]
        q, n = self.P[i], self.Nn[i]
        return ((X - q) * n).sum(1), n, q

    def push_out(self, X, clearance, iters=3, mask=None):
        X = X.copy()
        clearance = np.broadcast_to(np.asarray(clearance, float), (len(X),)).copy()
        for _ in range(iters):
            s, n, _ = self.query(X)
            bad = s < clearance
            if mask is not None:
                bad &= mask
            if not bad.any():
                break
            X[bad] += n[bad] * (clearance[bad] - s[bad])[:, None]
        return X


class QuadMesh:
    """Welded quads. P (n,3) current shape, B (n,3) body anchor (skin point it came from), D (n,nb) dense weights."""

    def __init__(self, P, Q, B=None, D=None):
        self.P = np.asarray(P, float)
        self.Q = np.asarray(Q, np.int64)
        self.B = self.P.copy() if B is None else np.asarray(B, float)
        self.D = D

    @staticmethod
    def from_body(base, face_ids, D):
        Qg = np.array([[vi for vi, _ in base.obj.faces[f]] for f in face_ids], np.int64)
        u, inv = np.unique(Qg, return_inverse=True)
        Q = inv.reshape(-1, 4)
        m = QuadMesh(base.VA[u], Q, base.VA[u], D[u])
        m.orig = u
        return m

    def edges(self):
        E = np.concatenate([self.Q[:, [0, 1]], self.Q[:, [1, 2]], self.Q[:, [2, 3]], self.Q[:, [3, 0]]])
        E.sort(1)
        return np.unique(E, axis=0)

    def boundary_mask(self):
        E = np.concatenate([self.Q[:, [0, 1]], self.Q[:, [1, 2]], self.Q[:, [2, 3]], self.Q[:, [3, 0]]])
        E.sort(1)
        u, c = np.unique(E, axis=0, return_counts=True)
        b = np.zeros(len(self.P), bool)
        b[u[c == 1].reshape(-1)] = True
        return b

    def normals(self):
        return quad_normals(self.P, self.Q)

    def subset(self, face_mask):
        Q = self.Q[face_mask]
        u, inv = np.unique(Q, return_inverse=True)
        m = QuadMesh(self.P[u], inv.reshape(-1, 4), self.B[u], None if self.D is None else self.D[u])
        for k, v in self.__dict__.items():
            if k not in ("P", "Q", "B", "D") and isinstance(v, np.ndarray) and len(v) == len(self.P):
                setattr(m, k, v[u])
        return m

    def subdivide(self):
        n = len(self.P)
        E = self.edges()
        eid = {(int(a), int(b)): n + i for i, (a, b) in enumerate(E)}
        nf = len(self.Q)
        fc = n + len(E) + np.arange(nf)

        def lerp(A):
            if A is None:
                return None
            return np.concatenate([A, 0.5 * (A[E[:, 0]] + A[E[:, 1]]), A[self.Q].mean(1)])

        newQ = np.zeros((nf * 4, 4), np.int64)
        Qa = self.Q
        ek = lambda a, b: eid[(min(a, b), max(a, b))]  # noqa: E731
        k = 0
        for f in range(nf):
            a, b, c, d = (int(x) for x in Qa[f])
            ab, bc, cd, da = ek(a, b), ek(b, c), ek(c, d), ek(d, a)
            ce = fc[f]
            newQ[k] = (a, ab, ce, da); newQ[k + 1] = (ab, b, bc, ce)
            newQ[k + 2] = (ce, bc, c, cd); newQ[k + 3] = (da, ce, cd, d)
            k += 4
        m = QuadMesh(lerp(self.P), newQ, lerp(self.B), lerp(self.D))
        return m

    def neighbors(self):
        E = self.edges()
        n = len(self.P)
        import scipy.sparse as sp
        A = sp.coo_matrix((np.ones(len(E) * 2), (np.r_[E[:, 0], E[:, 1]], np.r_[E[:, 1], E[:, 0]])), shape=(n, n)).tocsr()
        deg = np.asarray(A.sum(1)).ravel()
        return A, deg

    def smooth(self, X, iters=5, lam=0.5, mu=-0.53, pin=None):
        """Taubin smoothing of an attribute (n,k) over the mesh graph; pin = bool mask of fixed vertices."""
        A, deg = self.neighbors()
        X = np.array(X, float)
        for _ in range(iters):
            for f in (lam, mu):
                L = (A @ X) / np.maximum(deg, 1)[:, None] - X
                if pin is not None:
                    L[pin] = 0
                X = X + f * L
        return X

    def lap_smooth(self, X, iters=5, lam=0.5, pin=None):
        A, deg = self.neighbors()
        X = np.array(X, float)
        for _ in range(iters):
            L = (A @ X) / np.maximum(deg, 1)[:, None] - X
            if pin is not None:
                L[pin] = 0
            X = X + lam * L
        return X

    def tris(self):
        return np.concatenate([self.Q[:, [0, 1, 2]], self.Q[:, [0, 2, 3]]])


def top4(D):
    """dense weights (n,nb) -> (J u16 (n,4), W f32 (n,4)) sorted descending, normalised."""
    J = np.argsort(-D, axis=1)[:, :4]
    W = np.take_along_axis(D, J, 1)
    W = np.maximum(W, 0)
    W = W / np.maximum(W.sum(1, keepdims=True), 1e-9)
    return J.astype(np.uint16), W.astype(np.float32)


def weights_from_body(base, D_body, X, k=6, body_ids=None):
    """Dense weights for arbitrary points X: inverse-distance blend of the k nearest skin vertices."""
    ids = base.verts("body") if body_ids is None else body_ids
    tree = cKDTree(base.VA[ids])
    d, i = tree.query(X, k=k)
    w = 1.0 / np.maximum(d, 1e-4) ** 2
    w /= w.sum(1, keepdims=True)
    return (D_body[ids][i] * w[:, :, None]).sum(1)


def sweep(C, Nrm, profile, closed=False, cap_ends=True):
    """Sweep a 2D profile [(across, up), ...] (closed polygon, CCW seen along +tangent) along curve C (n,3) with
    surface normals Nrm (n,3). Returns (P (n*m,3), Q quads, uv (n*m,2) metres: u = along curve, v = profile arc),
    with seam duplicates for UV wrap (closed curve => an extra column; profile loop => an extra row)."""
    C = np.asarray(C, float)
    n = len(C)
    T = np.gradient(C, axis=0)
    if closed:
        T = np.roll(C, -1, 0) - np.roll(C, 1, 0)
    T = normalize(T)
    Nn = normalize(Nrm - T * (Nrm * T).sum(1, keepdims=True))
    Bn = np.cross(Nn, T)  # across
    prof = np.asarray(profile, float)
    m = len(prof)
    prof_c = np.vstack([prof, prof[:1]])  # close the loop (seam row)
    seg = np.linalg.norm(np.diff(prof_c, axis=0), axis=1)
    pv = np.r_[0, np.cumsum(seg)]
    cols = n + 1 if closed else n
    idx = np.r_[np.arange(n), [0]] if closed else np.arange(n)
    s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(C[idx], axis=0), axis=1))]
    P = (C[idx][:, None, :] + Bn[idx][:, None, :] * prof_c[None, :, 0:1] + Nn[idx][:, None, :] * prof_c[None, :, 1:2])
    UV = np.stack(np.broadcast_arrays(s[:, None], pv[None, :]), -1)
    rows = m + 1
    vid = np.arange(cols * rows).reshape(cols, rows)
    Q = []
    for i in range(cols - 1):
        for j in range(rows - 1):
            Q.append((vid[i, j], vid[i + 1, j], vid[i + 1, j + 1], vid[i, j + 1]))
    P = P.reshape(-1, 3)
    UV = UV.reshape(-1, 2)
    weld = np.r_[np.arange(n), [0]][:, None] * m + np.r_[np.arange(m), [0]][None, :] if closed else \
        np.arange(n)[:, None] * m + np.r_[np.arange(m), [0]][None, :]
    weld = weld.reshape(-1)
    Q = np.array(Q, np.int64)
    capT = []
    if cap_ends and not closed:
        # fan caps at both ends (own vertices so the end face is flat-shaded)
        for end, sgn in ((0, -1), (cols - 1, 1)):
            ring = vid[end, :m]
            c = P[ring].mean(0)
            base_i = len(P)
            P = np.vstack([P, P[ring], c[None]])
            UV = np.vstack([UV, prof[:, :2] - prof[:, :2].mean(0) + s[end], [[s[end], 0]]])
            weld = np.r_[weld, weld.max() + 1 + np.arange(m + 1)]
            for j in range(m):
                a, b = base_i + j, base_i + (j + 1) % m
                capT.append((base_i + m, b, a) if sgn > 0 else (base_i + m, a, b))
    return P, Q, UV, weld, np.array(capT, np.int64).reshape(-1, 3)


def rounded_rect(w, h, r, nr=3, center_up=0.0):
    """CCW (across, up) rounded rectangle centred on across=0, from up=0 to up=h (r = corner radius)."""
    pts = []
    cs = [(w / 2 - r, h - r, 0), (-w / 2 + r, h - r, 90), (-w / 2 + r, r, 180), (w / 2 - r, r, 270)]
    for cx, cy, a0 in cs:
        for k in range(nr + 1):
            a = np.radians(a0 + 90 * k / nr)
            pts.append((cx + r * np.cos(a), cy + r * np.sin(a) + center_up))
    return np.array(pts)


def make_part(name, material, P, tris, UV, D, weld=None, double_sided=False):
    J, W = top4(D)
    return Part(name, material, P, UV, tris, J, W, weld=weld, double_sided=double_sided)


def split_by_corner_uv(P, Q, cornerUV, D, extra=None):
    """Quads with per-corner UVs -> unique (vertex, uv) local vertices. Returns P, tris, UV, D, weld, extra."""
    n4 = Q.reshape(-1)
    uvq = np.round(cornerUV.reshape(-1, 2), 5)
    key = np.concatenate([n4[:, None].astype(float), uvq], 1)
    u, first, inv = np.unique(key, axis=0, return_index=True, return_inverse=True)
    inv = inv.reshape(-1)
    vid = n4[first]
    Ql = inv.reshape(-1, 4)
    tris = np.concatenate([Ql[:, [0, 1, 2]], Ql[:, [0, 2, 3]]])
    ex = None if extra is None else extra[vid]
    return P[vid], tris, cornerUV.reshape(-1, 2)[first], D[vid], vid, ex
