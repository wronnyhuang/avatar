"""junor — generic UV rasteriser (pure numpy): triangles with per-corner UVs -> per-texel triangle id + barycentrics.

R = raster(uv_tris (nT,3,2), n)   row 0 = TOP = v 1
R.interp(attr_corners (nT,3,C)) -> (n,n,C); R.mask (n,n)
dilate(img, mask) fills empty texels with the nearest covered texel.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage


class Raster:
    pass


def raster(uvt, W, Hh=None):
    Hh = W if Hh is None else Hh
    U = np.stack([uvt[..., 0] * W, (1 - uvt[..., 1]) * Hh], -1)
    tri = np.full((Hh, W), -1, np.int32)
    bary = np.zeros((Hh, W, 3), np.float32)
    x0 = np.clip(np.floor(U[..., 0].min(1) - 0.5).astype(int), 0, W - 1)
    x1 = np.clip(np.ceil(U[..., 0].max(1) - 0.5).astype(int), 0, W - 1)
    y0 = np.clip(np.floor(U[..., 1].min(1) - 0.5).astype(int), 0, Hh - 1)
    y1 = np.clip(np.ceil(U[..., 1].max(1) - 0.5).astype(int), 0, Hh - 1)
    w, h = x1 - x0 + 1, y1 - y0 + 1
    cnt = w * h
    groups, chunk, acc = [], [], 0
    for i in range(len(U)):
        chunk.append(i)
        acc += cnt[i]
        if acc > 4_000_000:
            groups.append(chunk)
            chunk, acc = [], 0
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
        Pq = np.stack([px + 0.5, py + 0.5], -1)
        a, b, cc = U[ti, 0], U[ti, 1], U[ti, 2]
        v0, v1, v2 = b - a, cc - a, Pq - a
        den = v0[:, 0] * v1[:, 1] - v1[:, 0] * v0[:, 1]
        den = np.where(np.abs(den) < 1e-12, 1e-12, den)
        l1 = (v2[:, 0] * v1[:, 1] - v1[:, 0] * v2[:, 1]) / den
        l2 = (v0[:, 0] * v2[:, 1] - v2[:, 0] * v0[:, 1]) / den
        l0 = 1 - l1 - l2
        ins = (l0 >= -1e-4) & (l1 >= -1e-4) & (l2 >= -1e-4)
        tri[py[ins], px[ins]] = ti[ins]
        bary[py[ins], px[ins]] = np.stack([l0, l1, l2], -1)[ins]
    R = Raster()
    R.tri, R.bary, R.mask, R.n = tri, bary, tri >= 0, W

    def interp(attr):
        attr = np.asarray(attr, np.float64)
        out = np.zeros((Hh, W, attr.shape[-1]))
        m = R.mask
        out[m] = (attr[tri[m]] * bary[m][..., None]).sum(1)
        return out
    R.interp = interp
    return R


def dilate(img, mask):
    if mask.all():
        return img
    _, (iy, ix) = ndimage.distance_transform_edt(~mask, return_indices=True)
    return img[iy, ix]
