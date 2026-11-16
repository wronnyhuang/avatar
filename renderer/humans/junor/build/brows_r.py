"""junor — PLACEHOLDER eyebrow strands (the HAIR piece owns the real groom): strands rooted in a tapered band
along the ICT Multi-PIE brow landmarks, combed up at the head of the brow and lateral along the body/tail, lying on the
skin (each point re-projected onto the head surface + a small lift).  Output = juno.asset/1 hair groups brows_l/brows_r
(JHR1 .bin, follow "surface", bound to the head submesh), written by write(out, head_part_lod0, head_part_lod1).
"""
from __future__ import annotations

import os

import numpy as np

from graft import closest_on_tris
from humans import asset_io

N_STRANDS = 1300
N_PTS = 8


def _proj(part, Q):
    t, b, _ = closest_on_tris(part.P, part.tris, Q)
    return (part.P[part.tris[t]] * b[:, :, None]).sum(1), t, b


def _normals_at(part, t, b):
    N = part.N if part.N is not None else part.welded_normals()
    n = (N[part.tris[t]] * b[:, :, None]).sum(1)
    return n / np.linalg.norm(n, axis=1, keepdims=True)


def strands(part, curve, rng, n=N_STRANDS):
    """curve: (5,3) brow landmarks inner -> outer."""
    seg = np.linalg.norm(np.diff(curve, axis=0), axis=1)
    cum = np.r_[0, np.cumsum(seg)] / seg.sum()
    s = np.sort(rng.random(n) ** 1.15)
    C = np.stack([np.interp(s, cum, curve[:, k]) for k in range(3)], 1)
    T = np.stack([np.interp(np.clip(s + 0.02, 0, 1), cum, curve[:, k]) for k in range(3)], 1) - \
        np.stack([np.interp(np.clip(s - 0.02, 0, 1), cum, curve[:, k]) for k in range(3)], 1)
    T /= np.linalg.norm(T, axis=1, keepdims=True)
    up = np.array([0.0, 1.0, 0.0])
    width = 0.0078 * (1 - s) + 0.0026 * s
    off = (rng.random(n) - 0.5) * width
    root, t, b = _proj(part, C + off[:, None] * up + np.array([0, -0.0025, 0.004]))
    nrm = _normals_at(part, t, b)
    # direction: up + slightly lateral at the head (s < .25), lateral + a little up/down along the body (toward the
    # brow's mid line), on the skin plane
    lat = T
    d = np.where((s < 0.18)[:, None], 0.6 * up + 0.7 * lat, lat + (0.25 - 0.9 * off / np.maximum(width, 1e-4))[:, None] * up * 0.6)
    d = d - nrm * (d * nrm).sum(1, keepdims=True)
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    L = (0.0062 * (1 - s) + 0.0055 * s) * (0.75 + 0.5 * rng.random(n))
    pts = [root]
    for k in range(1, N_PTS):
        q = root + d * (L * k / (N_PTS - 1))[:, None]
        qp, tq, bq = _proj(part, q)
        nq = _normals_at(part, tq, bq)
        pts.append(qp + nq * (0.00025 + 0.0006 * k / (N_PTS - 1)))
    pos = np.stack(pts, 1)
    return pos, t, b


def write(out, parts0, parts1, landmarks, seed=5):
    """-> hair_groups meta dict; writes hair/<lod>_brows_<s>.bin"""
    os.makedirs(os.path.join(out, "hair"), exist_ok=True)
    groups = {}
    for side, key in (("l", "brow_l"), ("r", "brow_r")):
        curve = np.asarray(landmarks[key], np.float64)
        if curve[0, 0] * curve[-1, 0] > 0 and abs(curve[0, 0]) > abs(curve[-1, 0]):
            curve = curve[::-1]          # inner -> outer
        meta = {"bone": "head", "follow": "surface", "surface_submesh": "head", "width_root_m": 0.0001,
                "width_tip_m": 3.5e-05, "color_root": [0.06, 0.036, 0.022], "color_tip": [0.13, 0.08, 0.05],
                "melanin": 0.85, "redness": 0.5, "roughness": 0.45, "groom": " HEAD placeholder brow",
                "lod": {}, "lod_strands": {}, "points": N_PTS}
        for lod, part, ns in (("lod0", parts0, N_STRANDS), ("lod1", parts1, N_STRANDS // 2)):
            rng = np.random.default_rng(seed + (side == "r"))
            pos, t, b = strands(part, curve, rng, ns)
            uv = (part.UV[part.tris[t]] * b[:, :, None]).sum(1)
            S_ = len(pos)
            fn = f"hair/{lod}_brows_{side}.bin"
            jit = 0.8 + 0.4 * rng.random(S_)
            w = np.stack([1.0e-4 * jit, 3.0e-5 * jit], 1).astype(np.float32)
            asset_io.write_hair(os.path.join(out, fn), pos, uv, w, rng.random(S_), np.arange(S_), tri=t, bary=b)
            meta["lod"][lod] = fn
            meta["lod_strands"][lod] = S_
            if lod == "lod0":
                meta["strands"] = S_
        groups[f"brows_{side}"] = meta
    return groups
