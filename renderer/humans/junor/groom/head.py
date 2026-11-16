"""HAIR — the head as seen in a BUILT asset (any juno.asset/1 build: juno.asset, junor.asset vN).

Everything the groom scripts need comes from the asset alone (no build-pipeline objects), so the groom re-runs on
every new head HEAD publishes:

    H = Head(asset_dir)            # lod0 head submesh, landmarks, derived head frame, collision samples
    H.O, H.u, H.eye_y ...          # skull centre (sphere fit of the cranium), unit = head_top.y - eye.y
    H.sd(X) -> (s, n)              # signed distance to the head skin (k-NN on dense samples) + outward normal
    H.sd_body(X)                   # same for neck / torso / clothes (collision of long hair)
    H.push(X, h)                   # push points out to >= h above the head skin
    H.sample(n, rng)               # area-uniform points on the head skin: (pts, nrm)
    bind(asset_dir, lod, roots)    # nearest head triangle of THAT lod -> tri (local), bary, uv, dist

Frame conventions (asset space, metres): +y up, +z forward, +x = her left (eye_l is at +x).
"""
from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np
from scipy.spatial import cKDTree

HERE = os.path.dirname(os.path.abspath(__file__))
RENDERER = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
if RENDERER not in sys.path:
    sys.path.insert(0, RENDERER)
from humans import asset_io  # noqa: E402


def nrm(v):
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-12)


def smoothstep(a, b, x):
    t = np.clip((np.asarray(x, float) - a) / (b - a), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def asset_json(asset_dir):
    p = os.path.join(asset_dir, "juno.asset.json")
    if os.path.exists(p):
        return p
    c = sorted(glob.glob(os.path.join(asset_dir, "*.asset.json")))
    if not c:
        raise FileNotFoundError(f"no *.asset.json in {asset_dir}")
    return c[0]


def load(asset_dir, lod="lod0"):
    return asset_io.load_asset(asset_dir, lod, hair=False, validate=False)


def submesh_tris(A, names=("head",), materials=("skin_head",)):
    """global vertex ids (T,3) of a submesh + its first local triangle id 0.. (index range order)."""
    for s in A.submeshes:
        if s.get("name") in names or s.get("material") in materials:
            lo, n = s["first_index"], s["index_count"]
            I = A.mesh["index"][lo:lo + n].reshape(-1, 3).astype(np.int64)
            v0, vc = s["first_vertex"], s["vertex_count"]
            if I.max() < vc and v0 > 0 and I.min() < v0:
                I = I + v0                          # local indices -> global
            return I, s
    raise KeyError(f"no submesh {names}/{materials}")


def tri_sample(V, Nv, T, n, rng):
    a, b, c = V[T[:, 0]], V[T[:, 1]], V[T[:, 2]]
    area = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
    t = rng.choice(len(T), n, p=area / area.sum())
    r1, r2 = rng.random(n), rng.random(n)
    s1 = np.sqrt(r1)
    bc = np.stack([1 - s1, s1 * (1 - r2), s1 * r2], 1)
    TT = T[t]
    P = (bc[:, :, None] * V[TT]).sum(1)
    N = nrm((bc[:, :, None] * Nv[TT]).sum(1))
    return P, N, t, bc


def sphere_fit(X):
    A = np.c_[2 * X, np.ones(len(X))]
    b = (X ** 2).sum(1)
    sol = np.linalg.lstsq(A, b, rcond=None)[0]
    c = sol[:3]
    r = float(np.sqrt(sol[3] + c @ c))
    return c, r


class Head:
    def __init__(self, asset_dir, lod="lod0", spacing=0.0009, seed=4557):
        rng = np.random.default_rng(seed)
        A = self.A = load(asset_dir, lod)
        self.dir = asset_dir
        V = self.V = A.mesh["position"].astype(np.float64)
        Nv = self.Nv = nrm(A.mesh["normal"].astype(np.float64))
        T, sub = submesh_tris(A)
        self.T, self.sub = T, sub
        lm = A.meta.get("landmarks", {})
        hv = V[np.unique(T)]
        self.hv = hv

        def L(k, d=None):
            v = lm.get(k)
            return np.asarray(v[:3], float) if isinstance(v, (list, tuple)) and len(v) >= 3 else d
        top = L("head_top", hv[np.argmax(hv[:, 1])])
        el, er = L("eye_l"), L("eye_r")
        if el is None or er is None:
            raise KeyError("asset landmarks need eye_l / eye_r")
        chin = L("chin", None)
        self.eye_l, self.eye_r, self.top = el, er, top
        self.eye = 0.5 * (el + er)
        self.eye_y = float(self.eye[1])
        self.u = float(top[1] - self.eye_y)                      # head unit: eye line -> crown
        if chin is None:
            chin = self.eye - np.array([0.0, 0.9 * self.u, -0.01])
        self.chin = chin
        self.mouth = L("mouth", self.eye + 0.6 * (chin - self.eye))
        self.nose = L("nose_tip", self.eye + 0.35 * (chin - self.eye) + np.array([0, 0, 0.03]))
        # skull centre: sphere fit of the cranium (above the brow, behind the face)
        cr = hv[(hv[:, 1] > self.eye_y + 0.25 * self.u) & (hv[:, 2] < self.eye[2] - 0.0 * self.u)]
        c, r = sphere_fit(cr)
        c[0] = 0.0
        self.O, self.R = c, r
        # ears: the outermost head points between the mouth and the brow, behind the eyes
        band = (hv[:, 1] > self.mouth[1]) & (hv[:, 1] < self.eye_y + 0.2 * self.u) & (hv[:, 2] < self.eye[2] - 0.02)
        self.ear = {}
        for sgn, nm in ((1, "l"), (-1, "r")):
            q = hv[band & (sgn * hv[:, 0] > 0)]
            xm = np.abs(q[:, 0]).max()
            e = q[np.abs(q[:, 0]) > xm - 0.008]
            ctr = e.mean(0)
            near = hv[(np.abs(hv[:, 0]) > xm - 0.016) & (sgn * hv[:, 0] > 0) & (np.abs(hv[:, 2] - ctr[2]) < 0.035)
                      & (np.abs(hv[:, 1] - ctr[1]) < 0.05)]
            self.ear[nm] = dict(c=ctr, top=float(near[:, 1].max()), bot=float(near[:, 1].min()),
                                back=float(near[:, 2].min()), front=float(near[:, 2].max()), x=float(xm))
        # dense collision samples of the head skin
        a, b, cc = V[T[:, 0]], V[T[:, 1]], V[T[:, 2]]
        area = 0.5 * np.linalg.norm(np.cross(b - a, cc - a), axis=1).sum()
        M = int(area / spacing ** 2)
        self.P, self.N, _, _ = tri_sample(V, Nv, T, M, rng)
        self.kd = cKDTree(self.P)
        self.area = area
        # body / clothes (neck, shoulders, collar): coarser samples
        bt = []
        for s in A.submeshes:
            nm = s.get("name", "")
            if nm == "body" or nm.startswith("cloth_"):
                lo, n = s["first_index"], s["index_count"]
                I = A.mesh["index"][lo:lo + n].reshape(-1, 3).astype(np.int64)
                if I.max() < s["vertex_count"] and s["first_vertex"] > 0 and I.min() < s["first_vertex"]:
                    I = I + s["first_vertex"]
                cen = V[I].mean(1)
                keep = (cen[:, 1] > self.chin[1] - 0.35) & (np.linalg.norm(cen - self.O, axis=1) < 0.45)
                bt.append(I[keep])
        if bt:
            BT = np.concatenate(bt)
            a, b, cc = V[BT[:, 0]], V[BT[:, 1]], V[BT[:, 2]]
            ar = 0.5 * np.linalg.norm(np.cross(b - a, cc - a), axis=1).sum()
            self.BP, self.BN, _, _ = tri_sample(V, Nv, BT, int(ar / 0.0025 ** 2), rng)
            self.bkd = cKDTree(self.BP)
        else:
            self.BP = None

    # ---------------------------------------------------------------- frame helpers
    def az_el(self, X):
        d = X - self.O
        r = np.linalg.norm(d, axis=-1)
        az = np.degrees(np.arctan2(d[..., 0], d[..., 2]))           # 0 = front, + = her left, +-180 = back
        el = np.degrees(np.arcsin(np.clip(d[..., 1] / np.maximum(r, 1e-9), -1, 1)))
        return az, el, r

    # ---------------------------------------------------------------- distances
    def _sd(self, kd, P, N, X, k, ub=0.03):
        """signed distance (k-NN plane average); points farther than ub from every sample get +ub (outside)."""
        sh = X.shape
        Xf = X.reshape(-1, 3)
        dd, i = kd.query(Xf, k=k, distance_upper_bound=ub, workers=-1)
        far = ~np.isfinite(dd[:, 0])
        i = np.where(np.isfinite(dd), i, i[:, :1])
        i[far] = 0
        n = nrm(N[i].mean(1))
        s = ((Xf[:, None, :] - P[i]) * N[i]).sum(2).mean(1)
        s[far] = 1.0                      # far = clearly outside (never pushed)
        return s.reshape(sh[:-1]), n.reshape(sh)

    def sd(self, X, k=4):
        return self._sd(self.kd, self.P, self.N, X, k)

    def sd_body(self, X, k=4):
        if self.BP is None:
            return np.full(X.shape[:-1], 1.0), np.zeros(X.shape)
        return self._sd(self.bkd, self.BP, self.BN, X, k)

    def push(self, X, h, iters=3, body=0.0):
        """push points to >= h above the head skin (and >= body above neck/clothes when body > 0)."""
        X = X.copy()
        h = np.broadcast_to(np.asarray(h, float), X.shape[:-1])
        for _ in range(iters):
            s, n = self.sd(X)
            m = s < h
            if m.any():
                X[m] += (h[m] - s[m])[:, None] * n[m]
            if body > 0.0 and self.BP is not None:
                s2, n2 = self.sd_body(X)
                m2 = (s2 < body) & (s2 > -0.03)
                if m2.any():
                    X[m2] += (body - s2[m2])[:, None] * n2[m2]
                m = m | m2
            if not m.any():
                break
        return X

    def sample(self, n, rng):
        P, N, t, bc = tri_sample(self.V, self.Nv, self.T, n, rng)
        return P, N


def bind(asset_dir, lod, roots, sub_names=("head",)):
    """nearest head triangle of `lod` for each root -> (tri local index, bary (S,3), uv (S,2), dist (S,))."""
    A = load(asset_dir, lod)
    T, _ = submesh_tris(A, sub_names)
    P = A.mesh["position"].astype(np.float64)
    UV = A.mesh["uv0"].astype(np.float64)
    cen = P[T].mean(1)
    _, cand = cKDTree(cen).query(roots, k=min(12, len(T)))
    best_t = np.zeros(len(roots), np.int64)
    best_b = np.zeros((len(roots), 3))
    best_d = np.full(len(roots), np.inf)
    for k in range(cand.shape[1]):
        t = cand[:, k]
        a, b, c = P[T[t, 0]], P[T[t, 1]], P[T[t, 2]]
        v0, v1, v2 = b - a, c - a, roots - a
        d00, d01, d11 = (v0 * v0).sum(1), (v0 * v1).sum(1), (v1 * v1).sum(1)
        d20, d21 = (v2 * v0).sum(1), (v2 * v1).sum(1)
        den = np.maximum(d00 * d11 - d01 * d01, 1e-18)
        v = (d11 * d20 - d01 * d21) / den
        w = (d00 * d21 - d01 * d20) / den
        bc = np.clip(np.stack([1 - v - w, v, w], 1), 0, 1)
        bc /= bc.sum(1, keepdims=True)
        q = bc[:, 0:1] * a + bc[:, 1:2] * b + bc[:, 2:3] * c
        d = np.linalg.norm(q - roots, axis=1)
        m = d < best_d
        best_d[m], best_t[m], best_b[m] = d[m], t[m], bc[m]
    uv = (best_b[:, :, None] * UV[T[best_t]]).sum(1)
    return best_t, best_b, uv, best_d


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


def write_groups(src, out, groups_raw, hair_render=None):
    """groups_raw {name: {"meta": {...}, "lod0": arrays, "lod1": arrays}} with arrays {pos (S,P,3), width (S,2),
    seed (S,), clump (S,)} -> a copy of `src` (hard links) at `out` with ONLY these hair groups (hair/*.bin rewritten,
    the json's hair_groups / lods[].hair / materials hair_* replaced; hair_render added when given)."""
    import shutil
    import subprocess
    if os.path.abspath(src) != os.path.abspath(out):
        if os.path.exists(out):
            shutil.rmtree(out)
        os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
        subprocess.check_call(["cp", "-al", src, out])
    hd = os.path.join(out, "hair")
    if os.path.exists(hd):
        shutil.rmtree(hd)
    os.makedirs(hd)
    groups = {}
    for g, h in groups_raw.items():
        meta = dict(h["meta"])
        meta.setdefault("surface_submesh", "head")
        meta.setdefault("bone", "head")
        meta.setdefault("follow", "bone")
        meta["lod"], meta["lod_strands"] = {}, {}
        for lod in ("lod0", "lod1"):
            arr = h[lod]
            pos = np.asarray(arr["pos"], np.float32)
            S_, P_ = pos.shape[:2]
            tri, bary, uv, dist = bind(out, lod, pos[:, 0].astype(np.float64), (meta["surface_submesh"],))
            fn = f"hair/{lod}_{g}.bin"
            asset_io.write_hair(os.path.join(out, fn), pos, uv, arr["width"], arr["seed"], arr["clump"],
                                tri=tri, bary=bary)
            meta["lod"][lod] = fn
            meta["lod_strands"][lod] = int(S_)
            meta["points"] = int(P_)
            if lod == "lod0":
                meta["strands"] = int(S_)
                meta["root_dist_max_mm"] = round(float(dist.max()) * 1000, 2)
        groups[g] = meta
    jp = asset_json(out)
    meta = json.load(open(jp))
    os.unlink(jp)                       # hard link -> write a fresh file
    meta["hair_groups"] = groups
    for l in meta["lods"]:
        l["hair"] = {g: gi["lod"][l["name"]] for g, gi in groups.items()}
    meta["materials"] = {k: v for k, v in meta.get("materials", {}).items() if not k.startswith("hair_")}
    for g, gi in groups.items():
        meta["materials"]["hair_" + g] = {"type": "hair", "params": {k: gi[k] for k in gi if k in
                                                                     ("color_root", "color_tip", "melanin", "roughness")}}
    if hair_render is not None:
        meta["hair_render"] = hair_render
    json.dump(meta, open(jp, "w"), indent=1)
    return groups
