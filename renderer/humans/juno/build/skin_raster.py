"""asset — skin helper: rasterize MH UV triangles into texture space (pure numpy) + seam dilation.

raster(base, faces, n) -> Raster with
  .tri  (n,n) int32   triangulated-face index per texel (-1 = empty)      (image row 0 = TOP = uv v 1)
  .bary (n,n,3) f32   barycentrics of the texel centre in that triangle
  .tv   (nT,3)  MH vertex ids per triangle,  .tt (nT,3) MH vt ids per triangle
  .interp(attr (nV,C)) -> (n,n,C)  per-vertex attribute interpolated into texture space (0 where empty)
  .mask (n,n) bool
dilate(img, mask) fills every empty texel with its nearest covered texel (seams never show under mip-mapping).
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage


class Raster:
    pass


def raster(base, faces, n):
    tv, tt = [], []
    for fi in faces:
        f = base.obj.faces[fi]
        vs = [v for v, _ in f]
        ts = [t for _, t in f]
        tv.append(vs[:3]); tt.append(ts[:3])
        if len(f) == 4:
            tv.append([vs[0], vs[2], vs[3]]); tt.append([ts[0], ts[2], ts[3]])
    tv = np.array(tv, np.int64)
    tt = np.array(tt, np.int64)
    vt = base.obj.vt
    # texel space: x = u*n, y = (1-v)*n ; texel centre at (c+.5, r+.5)
    U = np.stack([vt[tt][..., 0] * n, (1 - vt[tt][..., 1]) * n], -1)  # (nT,3,2)
    tri = np.full((n, n), -1, np.int32)
    bary = np.zeros((n, n, 3), np.float32)
    x0 = np.clip(np.floor(U[..., 0].min(1) - 0.5).astype(int), 0, n - 1)
    x1 = np.clip(np.ceil(U[..., 0].max(1) - 0.5).astype(int), 0, n - 1)
    y0 = np.clip(np.floor(U[..., 1].min(1) - 0.5).astype(int), 0, n - 1)
    y1 = np.clip(np.ceil(U[..., 1].max(1) - 0.5).astype(int), 0, n - 1)
    w = x1 - x0 + 1
    h = y1 - y0 + 1
    cnt = w * h
    order = np.arange(len(tv))
    chunk = []
    acc = 0
    groups = []
    for i in order:
        chunk.append(i)
        acc += cnt[i]
        if acc > 4_000_000:
            groups.append(chunk); chunk = []; acc = 0
    if chunk:
        groups.append(chunk)
    for g in groups:
        g = np.array(g)
        c = cnt[g]
        ti = np.repeat(g, c)
        start = np.repeat(np.cumsum(c) - c, c)
        k = np.arange(c.sum()) - start
        px = x0[ti] + k % w[ti]
        py = y0[ti] + k // w[ti]
        P = np.stack([px + 0.5, py + 0.5], -1)
        a, b, cc = U[ti, 0], U[ti, 1], U[ti, 2]
        v0, v1, v2 = b - a, cc - a, P - a
        den = v0[:, 0] * v1[:, 1] - v1[:, 0] * v0[:, 1]
        den = np.where(np.abs(den) < 1e-12, 1e-12, den)
        l1 = (v2[:, 0] * v1[:, 1] - v1[:, 0] * v2[:, 1]) / den
        l2 = (v0[:, 0] * v2[:, 1] - v2[:, 0] * v0[:, 1]) / den
        l0 = 1 - l1 - l2
        eps = -1e-4
        ok = (l0 >= eps) & (l1 >= eps) & (l2 >= eps)
        tri[py[ok], px[ok]] = ti[ok]
        bary[py[ok], px[ok]] = np.stack([l0, l1, l2], -1)[ok]
    R = Raster()
    R.n, R.tri, R.bary, R.tv, R.tt = n, tri, bary, tv, tt
    R.mask = tri >= 0
    R.faces = np.asarray(faces)

    def interp(attr):
        attr = np.asarray(attr, np.float32)
        if attr.ndim == 1:
            attr = attr[:, None]
        out = np.zeros((n, n, attr.shape[1]), np.float32)
        m = R.mask
        t = tri[m]
        bb = bary[m]
        out[m] = (attr[tv[t]] * bb[..., None]).sum(1)
        return out

    R.interp = interp
    return R


def dilate(img, mask):
    """nearest-covered fill for every empty texel."""
    _, (iy, ix) = ndimage.distance_transform_edt(~mask, return_indices=True)
    return img[iy, ix]
