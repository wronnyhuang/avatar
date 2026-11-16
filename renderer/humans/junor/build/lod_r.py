"""head-lod — lod1 decimation of junor Parts (pure numpy, build-time only).

Half-edge-collapse quadric decimation (Garland-Heckbert error, vertex v collapses ONTO a neighbour u), so the decimated
Part is a SUBSET of the original vertices: UVs, skin weights, normals, weld ids and every blendshape delta are copied from
the kept vertices unchanged (no re-projection, every shape keeps working on lod1).

Shape-aware: the quadrics are summed over the rest pose plus the Part's strongest blendshapes posed at 1, so regions that
deform (lips, lids, cheeks) keep the vertices they need to deform.

Locked (never removed): every vertex on a border of the SPLIT mesh — that is the real mesh borders (neck weld, inner lips,
garment hems, holes under clothes) AND every UV seam (the Part's vertices are split at seams, so seams are borders) — plus
non-manifold edges and an optional caller mask (eye zones). Rejected collapses: triangle flips (rest pose and every
sampled pose), UV-orientation flips, slivers.

    p1 = decimate(part, ratio=0.5, lock=mask, n_shapes=8)     -> new Part (same name/material/flags)
"""
from __future__ import annotations

import heapq

import numpy as np


def _plane_quadrics(P, T):
    """(nV, 4, 4) sum of the (unweighted, unit-normal) plane quadrics of each vertex's triangles."""
    a, b, c = P[T[:, 0]], P[T[:, 1]], P[T[:, 2]]
    n = np.cross(b - a, c - a)
    ln = np.linalg.norm(n, axis=1, keepdims=True)
    n = n / np.maximum(ln, 1e-20)
    d = -(n * a).sum(1, keepdims=True)
    p = np.concatenate([n, d], 1)
    K = p[:, :, None] * p[:, None, :]
    Q = np.zeros((len(P), 4, 4))
    for k in range(3):
        np.add.at(Q, T[:, k], K)
    return Q


def _tri_normal(a, b, c):
    n = np.cross(b - a, c - a)
    ln = np.linalg.norm(n)
    return n / ln if ln > 1e-20 else None


def _pt_tri_dist(p, a, b, c):
    """distance from points p (K,3) to triangle abc (Ericson's closest point, vectorised over p)."""
    ab, ac = b - a, c - a
    ap = p - a
    d1, d2 = ap @ ab, ap @ ac
    bp = p - b
    d3, d4 = bp @ ab, bp @ ac
    cp = p - c
    d5, d6 = cp @ ab, cp @ ac
    va = d3 * d6 - d5 * d4
    vb = d5 * d2 - d1 * d6
    vc = d1 * d4 - d3 * d2
    den = np.where(np.abs(va + vb + vc) < 1e-30, 1e-30, va + vb + vc)
    v = vb / den
    w = vc / den
    q = a + v[:, None] * ab + w[:, None] * ac
    # region fallbacks: clamp to the closest edge point when outside
    out = (va < 0) | (vb < 0) | (vc < 0)
    if out.any():
        best = np.full(len(p), np.inf)
        qq = q.copy()
        for x, y in ((a, b), (b, c), (c, a)):
            e = y - x
            t = np.clip(((p - x) @ e) / max(float(e @ e), 1e-30), 0, 1)
            r = x + t[:, None] * e
            dd = np.linalg.norm(p - r, axis=1)
            m = out & (dd < best)
            best[m] = dd[m]
            qq[m] = r[m]
        q = np.where(out[:, None], qq, q)
    return np.linalg.norm(p - q, axis=1)


def decimate(part, ratio=0.5, target=None, lock=None, n_shapes=8, max_err=None, min_angle_dot=0.25, max_dev=None,
             log=None):
    """max_dev (m): reject a collapse when any original vertex merged into the fan (incl. v) ends farther than max_dev
    from the new fan of u (a one-sided Hausdorff bound; used on layered cloth so an outer layer cannot sink into the
    one under it)."""
    from parts import Part
    P = np.asarray(part.P, np.float64)
    UV = np.asarray(part.UV, np.float64)
    T = np.asarray(part.tris, np.int64).copy()
    nV, nF = len(P), len(T)
    target = int(target if target is not None else round(nF * ratio))
    if target >= nF:
        return part
    # ---- poses: rest + strongest shapes
    sh = sorted(part.shapes.items(), key=lambda kv: -float(np.linalg.norm(kv[1][1], axis=1).sum()))[:n_shapes]
    poses = [P]
    for _, (li, dp) in sh:
        Pp = P.copy()
        Pp[li] += dp
        poses.append(Pp)
    Ps = np.stack(poses)                                    # (npose, nV, 3)
    Q = np.stack([_plane_quadrics(Pp, T) for Pp in poses])  # (npose, nV, 4, 4)
    Ph = np.concatenate([Ps, np.ones(Ps.shape[:2] + (1,))], 2)
    # ---- topology
    e = np.sort(np.concatenate([T[:, [0, 1]], T[:, [1, 2]], T[:, [2, 0]]]), 1)
    ek, ec = np.unique(e, axis=0, return_counts=True)
    locked = np.zeros(nV, bool)
    locked[ek[ec != 2].reshape(-1)] = True                  # borders (incl. UV seams of the split mesh) + non-manifold
    if lock is not None:
        locked |= np.asarray(lock, bool)
    vt = [set() for _ in range(nV)]
    for fi, t in enumerate(T):
        for v in t:
            vt[v].add(fi)
    alive_f = np.ones(nF, bool)
    alive_v = np.ones(nV, bool)
    ver = np.zeros(nV, np.int64)
    owned = [[v] for v in range(nV)] if max_dev else None

    def nbrs(v):
        s = set()
        for f in vt[v]:
            s.update(T[f])
        s.discard(v)
        return s

    def cost(v, u):
        q = Q[:, v] + Q[:, u]
        x = Ph[:, u]
        return float(np.einsum("pi,pij,pj->", x, q, x))

    heap = []

    def push(v):
        if locked[v] or not alive_v[v]:
            return
        for u in nbrs(v):
            heapq.heappush(heap, (cost(v, u), int(v), int(u), int(ver[v]), int(ver[u])))

    for v in range(nV):
        push(v)
    nf = nF
    ncol = 0
    while heap and nf > target:
        c, v, u, vv, vu = heapq.heappop(heap)
        if not (alive_v[v] and alive_v[u]) or ver[v] != vv or ver[u] != vu:
            continue
        if max_err is not None and c > max_err:
            break
        Nv, Nu = nbrs(v), nbrs(u)
        if u not in Nv:
            continue
        shared = [f for f in vt[v] if u in T[f]]
        if len(shared) != 2 or len(Nv & Nu) != 2:           # link condition (v interior => 2 shared tris)
            ver[v] += 1
            continue
        ok = True
        moved = [f for f in vt[v] if u not in T[f]]
        for f in moved:
            t = T[f]
            tn = np.where(t == v, u, t)
            for Pp in Ps:
                n0 = _tri_normal(Pp[t[0]], Pp[t[1]], Pp[t[2]])
                n1 = _tri_normal(Pp[tn[0]], Pp[tn[1]], Pp[tn[2]])
                if n0 is None or n1 is None or float(n0 @ n1) < min_angle_dot:
                    ok = False
                    break
            if not ok:
                break
            a0 = np.cross(np.r_[UV[t[1]] - UV[t[0]], 0], np.r_[UV[t[2]] - UV[t[0]], 0])[2]
            a1 = np.cross(np.r_[UV[tn[1]] - UV[tn[0]], 0], np.r_[UV[tn[2]] - UV[tn[0]], 0])[2]
            if a0 * a1 <= 0:
                ok = False
                break
            # sliver guard (rest pose): min edge / longest edge
            pa, pb, pc = P[tn[0]], P[tn[1]], P[tn[2]]
            ar = np.linalg.norm(np.cross(pb - pa, pc - pa))
            lmax = max(np.linalg.norm(pb - pa), np.linalg.norm(pc - pb), np.linalg.norm(pa - pc))
            if ar < 0.08 * lmax * lmax:
                ok = False
                break
        if ok and max_dev:
            fan = [np.where(T[f] == v, u, T[f]) for f in moved] + [T[f] for f in vt[u] if f not in shared]
            pts = P[owned[v] + owned[u]]
            dmin = np.full(len(pts), np.inf)
            for t in fan:
                dmin = np.minimum(dmin, _pt_tri_dist(pts, P[t[0]], P[t[1]], P[t[2]]))
            ok = float(dmin.max()) <= max_dev
        if not ok:
            continue                                         # this (v,u) stays invalid until a neighbour changes
        for f in shared:
            alive_f[f] = False
            for w in T[f]:
                vt[w].discard(f)
            nf -= 1
        for f in moved:
            T[f][T[f] == v] = u
            vt[u].add(f)
        vt[v].clear()
        alive_v[v] = False
        if owned is not None:
            owned[u] += owned[v]
            owned[v] = []
        Q[:, u] += Q[:, v]
        ncol += 1
        ver[u] += 1
        for w in Nv | Nu:
            ver[w] += 1
        push(u)
        for w in nbrs(u):
            push(w)
    # ---- rebuild the Part on the kept vertices
    T1 = T[alive_f]
    used = np.unique(T1)
    remap = -np.ones(nV, np.int64)
    remap[used] = np.arange(len(used))
    shapes = {}
    for sn, (li, dp) in part.shapes.items():
        m = remap[li] >= 0
        if m.any():
            shapes[sn] = (remap[li[m]], dp[m])
    N = None if part.N is None else np.asarray(part.N)[used]
    q = Part(part.name, part.material, P[used], UV[used], remap[T1], np.asarray(part.J)[used], np.asarray(part.W)[used],
             N=N, shapes=shapes, weld=np.asarray(part.weld)[used], blend=part.blend, double_sided=part.double_sided,
             orig=None if part.orig is None else np.asarray(part.orig)[used])
    if log:
        log(f"lod1 {part.name}: tris {nF} -> {len(T1)}, verts {nV} -> {len(used)}, locked {int(locked.sum())}, "
            f"poses {len(poses)}")
    return q


# lod1 tris targets as a fraction of lod0 (parts not listed are kept; the eyeballs/corneas come from eye_parts' lod1)
LOD1_RATIO = {"head": 0.55, "body": 0.5, "mouth_interior": 0.5, "teeth_upper": 0.35, "teeth_lower": 0.35,
              "gums_upper": 0.45, "gums_lower": 0.45, "cloth_cardigan": 0.6, "cloth_cardigan_rib": 0.5,
              "cloth_buttons": 0.5, "cloth_blouse": 0.45, "cloth_trousers": 0.5, "shoes": 0.4, "shoes_sole": 0.45}


LAYERED = ("cloth_", "shoes")
MOUTH_LOCK = (0.036, 0.024, 0.03)    # half-extents x, y (ellipse) and depth behind the mouth landmark, m


def lod1_parts(parts, swap, eyes=None, eye_lock_R=1.5, layer_lock_m=0.006, layer_dev_m=0.0006, mouth=None, log=print):
    """parts (lod0 order) -> lod1 parts. swap: {name: replacement Part} (eye_parts' low-res eyeballs/corneas).
    eyes: {side: (centre, radius)} -> head vertices within eye_lock_R * radius of an eye centre are locked (lids, lid
    margin under the occlusion/tearline strips, caruncle seat stay exactly lod0)."""
    out = []
    for p in parts:
        if p.name in swap:
            out.append(swap[p.name])
            continue
        r = LOD1_RATIO.get(p.name)
        if r is None:
            out.append(p)
            continue
        lock = None
        if p.name.startswith(LAYERED) and layer_lock_m > 0:
            # layered cloth: vertices within layer_lock_m of another garment / the skin stay (an outer layer's chords
            # sink into the layer under it -> the blouse/cardigan showed through the rib band at the shoulders)
            from scipy.spatial import cKDTree
            oth = [q.P for q in parts if q is not p and (q.name.startswith(LAYERED) or q.name in ("body", "head"))]
            dd, _ = cKDTree(np.concatenate(oth)).query(p.P)
            lock = dd < layer_lock_m
        if p.name == "head" and eyes:
            lock = np.zeros(len(p.P), bool)
            for c, R in eyes.values():
                lock |= np.linalg.norm(p.P - np.asarray(c)[None], axis=1) < eye_lock_R * R
            if mouth is not None:     # lips + corners + the nasolabial start: shading of speech shapes stays lod0
                m = np.asarray(mouth)
                q = p.P - m[None]
                lock |= ((q[:, 0] / MOUTH_LOCK[0]) ** 2 + (q[:, 1] / MOUTH_LOCK[1]) ** 2 < 1) & (q[:, 2] > -MOUTH_LOCK[2])
        out.append(decimate(p, ratio=r, lock=lock, max_dev=layer_dev_m if p.name.startswith(LAYERED) else None, log=log))
    return out
