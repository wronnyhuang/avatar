"""junor — the realistic HEAD from ICT-FaceKit (USC ICT Vision & Graphics Lab, MIT licence): light-stage scan
derived topology + identity PCA + 53 ARKit-style expression shapes (also scan derived).

    H = ICTHead(npz, identity_coeffs)       # npz = ict_io.py cache (T/asset-src/ict.npz)
    H.place(scale, offset)                  # cm -> asset metres (+y up, +z forward, +x = her LEFT, same as ICT)
    H.P (N,3) metres; H.expr {ICT name: (N,3) metres}; H.region(name) -> vertex ids; H.faces(region) -> (F, FT)

Regions follow the ICT README vertex table. The skin (Face + Head and Neck + mouth socket + eye sockets) is one
connected surface whose only open boundary is the neck loop (H.neck_loop, ordered).
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np

RANGES = {  # README "Face Model Topology" (inclusive ranges)
    "face": (0, 9408), "headneck": (9409, 11247), "mouth_socket": (11248, 13293), "eye_socket_l": (13294, 13677),
    "eye_socket_r": (13678, 14061), "gums_tongue": (14062, 17038), "teeth": (17039, 21450), "eyeball_l": (21451, 23020),
    "eyeball_r": (23021, 24590), "lacrimal_l": (24591, 24794), "lacrimal_r": (24795, 24998), "eyeblend_l": (24999, 25022),
    "eyeblend_r": (25023, 25046), "occlusion_l": (25047, 25198), "occlusion_r": (25199, 25350),
    "lashes_l": (25351, 26034), "lashes_r": (26035, 26718),
    "sclera_l": (21451, 22220), "iris_l": (22221, 23020), "sclera_r": (23021, 23790), "iris_r": (23791, 24590),
    "teeth_upper": (17039, 19246), "teeth_lower": (19247, 21450),
}
SKIN_REGIONS = ("face", "headneck", "mouth_socket", "eye_socket_l", "eye_socket_r")


def arkit_name(n):
    if n.endswith("_L"):
        return n[:-2] + "Left"
    if n.endswith("_R"):
        return n[:-2] + "Right"
    return n


class ICTHead:
    def __init__(self, npz_path, coeffs=None, sym=0.0):
        D = np.load(npz_path)
        self.V0 = D["V"].astype(np.float64)
        c = np.zeros(D["ident"].shape[0]) if coeffs is None else np.asarray(coeffs, np.float64)
        self.Vcm = self.V0 + np.tensordot(c, D["ident"].astype(np.float64), 1)
        if sym:   # pull the identity toward its mirror image (the PCA blend carried ~1 mm of lopsidedness)
            from scipy.spatial import cKDTree
            _, j = cKDTree(self.V0).query(self.V0 * [-1, 1, 1])
            Dl = self.Vcm - self.V0
            self.Vcm = self.V0 + Dl - 0.5 * sym * (Dl - Dl[j] * [-1, 1, 1])
        self.VT = D["VT"].astype(np.float64)
        self.F, self.FT, self.M = D["F"], D["FT"], D["M"]
        self.mats = [str(m) for m in D["mats"]]
        self.expr_cm = {str(n): D["expr"][i].astype(np.float64) for i, n in enumerate(D["expr_names"])}
        self.N = len(self.V0)
        self.place(0.01, np.zeros(3))
        self._loop()

    def place(self, scale, offset):
        self.scale, self.offset = float(scale), np.asarray(offset, np.float64)
        self.P = self.Vcm * self.scale + self.offset
        self.expr = {k: v * self.scale for k, v in self.expr_cm.items()}

    def region(self, name):
        a, b = RANGES[name]
        return np.arange(a, b + 1)

    def region_mask(self, *names):
        m = np.zeros(self.N, bool)
        for n in names:
            a, b = RANGES[n]
            m[a:b + 1] = True
        return m

    def faces(self, *names):
        """faces whose first vertex is in the regions -> (F (n,4) with -1 pad, FT)."""
        m = self.region_mask(*names)
        sel = m[self.F[:, 0]]
        return self.F[sel], self.FT[sel]

    def skin_faces(self):
        return self.faces(*SKIN_REGIONS)

    def _loop(self):
        F, _ = self.skin_faces()
        E = defaultdict(int)
        for f in F:
            f = [x for x in f if x >= 0]
            for i in range(len(f)):
                a, b = f[i], f[(i + 1) % len(f)]
                E[(min(a, b), max(a, b))] += 1
        adj = defaultdict(list)
        for (a, b), n in E.items():
            if n == 1:
                adj[a].append(b)
                adj[b].append(a)
        start = next(iter(adj))
        lp, prev = [start], None
        cur = start
        while True:
            nx = [n for n in adj[cur] if n != prev]
            if not nx or nx[0] == start:
                break
            prev, cur = cur, nx[0]
            lp.append(cur)
        assert len(lp) == len(adj), "skin has more than one open boundary"
        self.neck_loop = np.array(lp)

    def eye_sphere(self, side):
        """least-squares sphere through the sclera vertices -> (centre, radius) in placed metres."""
        P = self.P[self.region(f"sclera_{side}")]
        A = np.c_[2 * P, np.ones(len(P))]
        b = (P * P).sum(1)
        x, *_ = np.linalg.lstsq(A, b, rcond=None)
        c = x[:3]
        r = float(np.sqrt(x[3] + c @ c))
        return c, r

    def rigid_fit(self, ids, delta):
        """best rigid (R, t) with R P + t ~ P + delta over vertex ids (Kabsch)."""
        P = self.P[ids]
        Q = P + delta[ids]
        pc, qc = P.mean(0), Q.mean(0)
        U, _, Vt = np.linalg.svd((P - pc).T @ (Q - qc))
        d = np.sign(np.linalg.det(Vt.T @ U.T))
        R = Vt.T @ np.diag([1, 1, d]) @ U.T
        return R, qc - R @ pc
