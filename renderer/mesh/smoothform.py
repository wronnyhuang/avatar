"""render8: the face as ONE smooth stylized form (judge diag5: "realistic detail on toon proportions").

The v7 head carries sculpted micro-forms (under-eye creases, nasolabial folds, a pinch between the brows) in its
GEOMETRY: with every normal/AO map off they still read at card size (ld8/d_face.png).  The shading normal of the
head-skin vertices is blended toward a mesh-smoothed normal at load time -- rest normals AND every blendshape's normal
deltas go through the same linear smoothing operator, so smile / jaw creases soften too.  Positions (silhouette,
shadow-map occluders, lip seam) are untouched: only the light response is simplified.

cfg (juno.json mesh.face.smooth_form): {"k": blend 0..1, "iters": smoothing iterations, "mats": [material ids]}
absent or k = 0 -> the normals are returned unchanged (bit-identical picture).
"""
import numpy as np


def _operator(pos, tri, verts):
    """Row-normalised uniform Laplacian (incl. self) over the POSITION-welded graph of `verts` (UV seams joined)."""
    import scipy.sparse as sp
    q = np.round(pos[verts]/1e-6).astype(np.int64)
    _, weld, inv = np.unique(q, axis=0, return_index=True, return_inverse=True)
    inv = inv.reshape(-1)
    nw = len(weld)
    loc = np.full(len(pos), -1, np.int64)
    loc[verts] = inv
    t = loc[tri]
    t = t[(t >= 0).all(1)]
    r = np.concatenate([t[:, 0], t[:, 1], t[:, 2], t[:, 1], t[:, 2], t[:, 0], np.arange(nw)])
    c = np.concatenate([t[:, 1], t[:, 2], t[:, 0], t[:, 0], t[:, 1], t[:, 2], np.arange(nw)])
    W = sp.csr_matrix((np.ones(len(r)), (r, c)), shape=(nw, nw))
    W.data[:] = 1.0
    d = np.asarray(W.sum(1)).reshape(-1)
    W = sp.diags(1.0/np.maximum(d, 1.0)) @ W
    return W, inv


def smooth_form(A, cfg):
    """cached wrapper (JUNO_FACE_CACHE dir, keyed by the asset geometry + shapes + cfg): the 90-iteration delta smoothing is
    ~2-3 s on lod0 cold, ~0.05 s from the cache."""
    import os, json, hashlib
    cfg = cfg or {}
    if float(cfg.get("k", 0.0)) <= 0.0:
        return None
    try:
        from eyesmouth import face_cache_name
        key = face_cache_name(A).replace(".pkl", "") + hashlib.sha1(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:10]
        d = os.environ.get("JUNO_FACE_CACHE", os.path.join(os.path.expanduser("~"), ".cache", "juno_face"))
        path = os.path.join(d, f"sf_{key}.npz")
    except Exception:
        path = None
    if path and os.path.exists(path):
        try:
            z = np.load(path)
            dns = {int(k[1:]): z[k] for k in z.files if k.startswith("s")}
            return z["n"], z["v"], dns
        except Exception:
            pass
    r = _smooth_form(A, cfg)
    if path and r is not None:
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            tmp = path + f".{os.getpid()}.tmp.npz"
            np.savez(tmp, n=r[0], v=r[1], **{f"s{k}": v for k, v in r[2].items()})
            os.replace(tmp, path)
        except Exception:
            pass
    return r


def _smooth_form(A, cfg):
    """-> (normals float32[N,3], verts int[V], {shape index: dn float32[V,3] over verts}) or None when off."""
    cfg = cfg or {}
    k = float(cfg.get("k", 0.0))
    if k <= 0.0:
        return None
    iters = int(cfg.get("iters", 12))
    diters = int(cfg.get("delta_iters", iters))      # the shapes' folds (smile cheek, squint) may want more
    mats = cfg.get("mats", ["skin_head"])
    m = A.mesh
    pos = m["position"].astype(np.float64)
    nrm = m["normal"].astype(np.float64)
    idx = np.asarray(m["index"]).reshape(-1)
    tris, verts = [], []
    for s in A.submeshes:
        if s["material"] in mats:
            t = idx[s["first_index"]:s["first_index"] + s["index_count"]].reshape(-1, 3)
            tris.append(t)
    if not tris:
        return None
    tri = np.concatenate(tris)
    verts = np.unique(tri)
    W, inv = _operator(pos, tri, verts)

    def smooth(x, n_it=None):           # x: [len(verts), 3] per original vertex -> smoothed, per original vertex
        nw = W.shape[0]
        y = np.zeros((nw, 3)); cnt = np.zeros(nw)
        np.add.at(y, inv, x); np.add.at(cnt, inv, 1.0)
        y /= np.maximum(cnt, 1.0)[:, None]
        for _ in range(iters if n_it is None else n_it):
            y = W @ y
        return y[inv]

    # blend weight: k, faded to 0 at the submesh border (the neck seam into skin_body stays continuous)
    nw = W.shape[0]
    border = np.zeros(nw)
    # border = welded vertices on an open edge of the selection
    t = inv[np.searchsorted(verts, tri)]
    e = np.sort(np.concatenate([t[:, [0, 1]], t[:, [1, 2]], t[:, [2, 0]]]), 1)
    ue, cnt = np.unique(e, axis=0, return_counts=True)
    bv = np.unique(ue[cnt == 1])
    border[bv] = 1.0
    for _ in range(max(iters//2, 2)):
        border = np.maximum(border, W @ border)
    keep = 1.0 - np.clip(border, 0.0, 1.0)
    # lips keep their own volume (lip_keep): vertices the lip-roll / pucker shapes move the most
    lk = float(cfg.get("lip_keep", 0.0))
    if lk > 0.0:
        sel0 = np.full(len(pos), -1, np.int64); sel0[verts] = np.arange(len(verts))
        lm = np.zeros(len(verts))
        for nm, sidx, dp, _ in A.shapes:
            if nm in ("mouthRollUpper", "mouthRollLower", "mouthPucker", "mouthFunnel"):
                loc = sel0[np.asarray(sidx)]; ok = loc >= 0
                d = np.linalg.norm(np.asarray(dp, np.float64), axis=1)[ok]
                if len(d) and d.max() > 0:
                    lm[loc[ok]] = np.maximum(lm[loc[ok]], d/d.max())
        lmw = np.zeros(nw); np.maximum.at(lmw, inv, lm)
        lmw = np.clip((lmw - 0.2)/0.35, 0.0, 1.0)
        for _ in range(3):
            lmw = W @ lmw
        keep = keep*(1.0 - lk*np.clip(lmw, 0.0, 1.0))
    wv = (k*keep)[inv][:, None]

    ns = smooth(nrm[verts])
    out = nrm.copy()
    n0 = nrm[verts]
    nb = (1.0 - wv)*n0 + wv*ns
    out[verts] = nb/np.maximum(np.linalg.norm(nb, axis=1, keepdims=True), 1e-8)
    # shape normal deltas: the posed normal is normalize(n_rest' + sum w dn'); with n' = (1-w) n + w S n and the
    # operator linear, dn' = (1-w) dn + w S dn (the normalisation of the rest normal is undone by its length)
    sel = np.full(len(pos), -1, np.int64); sel[verts] = np.arange(len(verts))
    ln = np.maximum(np.linalg.norm(nb, axis=1), 1e-8)[:, None]
    dns = {}
    cols, sis = [], []
    for si, (_, sidx, _, dn) in enumerate(A.shapes):
        loc = sel[np.asarray(sidx)]
        ok = loc >= 0
        if not ok.any():
            continue
        full = np.zeros((len(verts), 3))
        full[loc[ok]] = np.asarray(dn, np.float64)[ok]
        cols.append(full); sis.append(si)
    if cols:
        X = np.concatenate(cols, 1)                       # every shape at once: one sparse x dense product per iteration
        nwv = W.shape[0]
        Y = np.zeros((nwv, X.shape[1])); cnt = np.zeros(nwv)
        np.add.at(Y, inv, X); np.add.at(cnt, inv, 1.0)
        Y /= np.maximum(cnt, 1.0)[:, None]
        for _ in range(diters):
            Y = W @ Y
        Y = Y[inv]
        for j, si in enumerate(sis):
            full = X[:, 3*j:3*j + 3]
            dns[si] = (((1.0 - wv)*full + wv*Y[:, 3*j:3*j + 3])/ln).astype(np.float32)   # every selected vertex
    return out.astype(np.float32), verts, dns
