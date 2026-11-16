"""asset-eyes2 — upper lash-line strip (submeshes lashline_l / lashline_r, material `lashline`, type "lashes").

The concept's thick dark upper lash line: a band on the upper lid skin, its lower edge ON the lid margin (over the lit
rim's top), ~1.2 mm tall over the iris, thin at the inner corner, thicker toward the outer third, then continuing PAST
the outer canthus as a small upswept WING that tapers to a point. Geometry, not paint: it rides every lid shape because
each strip vertex is bound (barycentric) to a lid-skin triangle of the fitted head and copies that triangle's shape
deltas (blink, squint, wide, look*, happy, correctives — the same rule eye_occl uses for the margin), lifted a constant
offset along the skin normal. Skinned 100 % to the head bone (like the lid strips). All sizes scale with the iris radius.

build(base, S, texdir) -> (parts, materials)
"""
from __future__ import annotations

import os

import numpy as np
from PIL import Image

from parts import Part

LASHLINE = {
    # asset-eyes4: NOT a liner any more (judge r4: "hard black lash/liner decals floating on the lids"). The lash
    # fringe is real clumped strands (hair.py lashes); this strip is only a faint soft root shadow ~0.4 mm tall hugging
    # the margin, no wing (eyes3 values in brackets).
    "h_mid": 0.07,       # band height over the iris (x iris_r) [.115]
    "h_in": 0.02,        # at the inner canthus [.025]
    "h_out": 0.10,       # at the outer canthus [.24]
    "wing_len": 0.05,    # vestigial (the strip just ends at the canthus) [.42]
    "wing_up_deg": 20.0, # [30]
    "over": 0.05,        # lower edge pushed this far (x iris_r) down over the margin rim [.11]
    "wing_blink_collapse": 0.92,  # eyes3: fraction of the wing folded back to the canthus at eyeBlink = 1
    "corner_blink_flatten": 0.7,  # eyes3: outer-corner band height flattened by this at eyeBlink = 1
    "offset_m": 0.00022, # lift along the skin normal (clears the subdivided lod0 surface)
    "alpha_max": 0.85,    # eyes4: peak opacity of the root shadow (soft both edges)
    "nu": 72, "nv": 5,
}


def _iris_r(base, side):
    df = getattr(base, "design_fit", None)
    if df and df.get("iris_radius"):
        return float(df["iris_radius"])
    return 0.47 * float(base.eye[side][1])


def _project(P, T, X, cand_k=16):
    """closest point on triangles (vertices P, tris T) for each point X -> (tri idx, bary, point)."""
    from scipy.spatial import cKDTree
    cen = P[T].mean(1)
    _, cand = cKDTree(cen).query(X, k=min(cand_k, len(T)))
    best_t = np.zeros(len(X), np.int64)
    best_b = np.zeros((len(X), 3))
    best_q = np.zeros((len(X), 3))
    best_d = np.full(len(X), np.inf)
    for k in range(cand.shape[1]):
        t = cand[:, k]
        a, b, c = P[T[t, 0]], P[T[t, 1]], P[T[t, 2]]
        v0, v1, v2 = b - a, c - a, X - a
        d00, d01, d11 = (v0 * v0).sum(1), (v0 * v1).sum(1), (v1 * v1).sum(1)
        d20, d21 = (v2 * v0).sum(1), (v2 * v1).sum(1)
        den = np.maximum(d00 * d11 - d01 * d01, 1e-18)
        v = (d11 * d20 - d01 * d21) / den
        w = (d00 * d21 - d01 * d20) / den
        bc = np.clip(np.stack([1 - v - w, v, w], 1), 0, 1)
        bc /= bc.sum(1, keepdims=True)
        q = bc[:, 0:1] * a + bc[:, 1:2] * b + bc[:, 2:3] * c
        d = np.linalg.norm(q - X, axis=1)
        m = d < best_d
        best_d[m], best_t[m], best_b[m], best_q[m] = d[m], t[m], bc[m], q[m]
    return best_t, best_b, best_q


def _project_z(P, T, X):
    """project points along the view axis (z) onto the front-most triangle under them (x,y inside the triangle);
    points with no triangle under them fall back to the closest point. -> (tri idx, bary, point)."""
    ti, bc, Q = _project(P, T, X)
    a, b, c = P[T[:, 0]], P[T[:, 1]], P[T[:, 2]]
    for i in range(0, len(X), 64):
        x = X[i:i + 64, None, :2]
        v0, v1, v2 = (b - a)[None, :, :2], (c - a)[None, :, :2], x - a[None, :, :2]
        den = v0[..., 0] * v1[..., 1] - v1[..., 0] * v0[..., 1]
        den = np.where(np.abs(den) < 1e-14, 1e-14, den)
        u = (v2[..., 0] * v1[..., 1] - v1[..., 0] * v2[..., 1]) / den
        w = (v0[..., 0] * v2[..., 1] - v2[..., 0] * v0[..., 1]) / den
        ins = (u >= -1e-6) & (w >= -1e-6) & (u + w <= 1 + 1e-6)
        z = a[None, :, 2] + u * (b - a)[None, :, 2] + w * (c - a)[None, :, 2]
        z = np.where(ins, z, -np.inf)
        j = np.argmax(z, 1)
        r = np.arange(len(j))
        ok = np.isfinite(z[r, j])
        for k in np.where(ok)[0]:
            jj = j[k]
            bb = np.array([1 - u[k, jj] - w[k, jj], u[k, jj], w[k, jj]])
            ti[i + k], bc[i + k] = jj, bb
            Q[i + k] = bb[0] * a[jj] + bb[1] * b[jj] + bb[2] * c[jj]
    return ti, bc, Q


def _margin_polyline(base, side, R, c):
    from juno_sculpt import lid_margins
    up, _ = lid_margins(base, side)
    q = base.VA[up] - c
    up = up[np.linalg.norm(q, axis=1) > 1.04 * R]                  # front row (no conjunctival tuck)
    sg = 1.0 if side == "l" else -1.0
    M = base.VA[up]
    o = np.argsort(sg * (M[:, 0] - c[0]))
    M = M[o]
    # drop duplicates (the margin has double vertices at UV seams) and smooth lightly
    keep = np.concatenate([[True], np.linalg.norm(np.diff(M, axis=0), axis=1) > 2e-4])
    M = M[keep]
    for _ in range(2):
        M[1:-1] = 0.25 * M[:-2] + 0.5 * M[1:-1] + 0.25 * M[2:]
    return M


def _resample(M, n):
    seg = np.linalg.norm(np.diff(M, axis=0), axis=1)
    L = np.concatenate([[0], np.cumsum(seg)])
    u = np.linspace(0, L[-1], n)
    return np.stack([np.interp(u, L, M[:, k]) for k in range(3)], 1), L[-1]


def strip_points(base, side, prm=None):
    """-> P (nu,nv,3) unprojected strip points, s (nu,) along-lid parameter (0 medial, 1 = where the upper lid turns
    down into the outer canthus, >1 = wing), v (nv,) 0..1 across (bottom edge .. top edge), prm.
    Bottom edge: the margin (pushed `over` down onto the rim) all the way into the outer canthus, then out to the wing
    tip. Top edge: margin + height up to the turn-down point, then straight (slightly bowed up) to the wing tip -> the
    outer corner fills as a wedge like a drawn cat-eye flick."""
    p = dict(LASHLINE, **(prm or {}))
    c, R = base.eye[side]
    c = np.asarray(c, float)
    ri = _iris_r(base, side)
    M = _margin_polyline(base, side, R, c)
    Mr, Llid = _resample(M, 200)
    x = np.abs(Mr[:, 0] - c[0])
    # turn-down point: outer third, where the margin starts to drop steeply toward the canthus
    slope = np.gradient(Mr[:, 1]) / np.maximum(np.gradient(x), 1e-9)
    cand = np.where((np.arange(200) > 120) & (slope < -1.3))[0]
    cut = int(cand[0]) if len(cand) else 190
    # up direction on the lid at each margin point
    T = np.gradient(Mr, axis=0)
    T /= np.linalg.norm(T, axis=1, keepdims=True)
    radial = (Mr - c) / np.linalg.norm(Mr - c, axis=1, keepdims=True)
    g = np.array([0, 1.0, 0])[None] * 0.7 + 0.3 * radial
    up = g - T * (g * T).sum(1, keepdims=True)
    up /= np.linalg.norm(up, axis=1, keepdims=True)
    sl = np.linspace(0, 1, 200) * (199 / max(cut, 1))          # 1 at the cut
    h = np.interp(np.clip(sl, 0, 1), [0, 0.18, 0.5, 0.82, 1.0],
                  [p["h_in"], 0.35 * p["h_mid"], p["h_mid"], p["h_out"], p["h_out"]]) * ri
    over = p["over"] * ri
    top_lid = Mr[:cut + 1] + h[:cut + 1, None] * up[:cut + 1]
    # wing: lateral (horizontal, outward from the eye) rotated up
    sg = 1.0 if side == "l" else -1.0
    a = np.radians(p["wing_up_deg"])
    wdir = np.array([sg * np.cos(a), np.sin(a), -0.25 * np.cos(a)])
    wdir /= np.linalg.norm(wdir)
    canth = Mr[-1]
    wl = p["wing_len"] * ri
    tip = canth + wl * wdir + np.array([0, 0.35 * h[cut], 0])
    tw = np.linspace(0, 1, 24)[1:]
    bow = (0.10 * wl * np.sin(np.pi * tw))[:, None] * np.array([0, 1.0, 0])
    top_w = top_lid[-1] + tw[:, None] * (tip - top_lid[-1]) + bow
    bot_w = canth + tw[:, None] * (tip - canth)
    top = np.concatenate([top_lid, top_w])
    for _ in range(60):     # a clean drawn curve: smooth the top edge (ends fixed)
        top[1:-1] = top[1:-1] + 0.5 * (0.5 * (top[:-2] + top[2:]) - top[1:-1])
    bot = np.concatenate([Mr - over * up, bot_w])
    nu, nv = p["nu"], p["nv"]
    tr, Lt = _resample(top, nu)
    br, Lb = _resample(bot, nu)
    v = np.linspace(0, 1, nv)
    P = br[:, None] + v[None, :, None] * (tr - br)[:, None]
    # along parameter: arc length of the top edge, 1 at the cut
    seg = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(tr, axis=0), axis=1))])
    Lcut = np.sum(np.linalg.norm(np.diff(top_lid, axis=0), axis=1))
    s = seg / max(Lcut, 1e-9)
    return P, s, v, p


def _skin_tris(base, side, R, c):
    """front lid/skin triangles around the eye (excluding the hidden inner lid tuck)."""
    VA = base.VA
    r = np.linalg.norm(VA - c, axis=1)
    ok = base.skin_mask & (r < 2.4 * R)
    tm = ok[base.tris].all(1) & (r[base.tris].max(1) > 1.03 * R)   # drop only the pure conjunctival-tuck triangles
    return base.tris[tm]


def build(base, S, texdir, prm=None):
    parts, mats = [], {}
    for side in ("l", "r"):
        c, R = base.eye[side]
        c = np.asarray(c, float)
        P0, s, v, p = strip_points(base, side, prm)
        ri = _iris_r(base, side)
        nu, nv = P0.shape[:2]
        X = P0.reshape(-1, 3)
        T = _skin_tris(base, side, R, c)
        VA = base.VA
        ti, bc, Q = _project_z(VA, T, X)
        tri = T[ti]
        nrm = np.cross(VA[tri[:, 1]] - VA[tri[:, 0]], VA[tri[:, 2]] - VA[tri[:, 0]])
        nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-12)
        # orient away from the eye centre
        nrm *= np.sign(((Q - c) * nrm).sum(1))[:, None]
        Pp = Q + p["offset_m"] * nrm
        # shapes: barycentric copy of the skin deltas of the bound triangle; each column then moves (mostly) with its
        # bottom vertex (the lash line on the margin), so a closing lid carries the band rigidly instead of shearing it
        # (the wing past the canthus was folding at blink 1)
        shapes = {}
        rigid = p.get("column_rigid", 0.85)
        # bottom row on the lid (s <= 1): deltas of the nearest two front-margin vertices (the lash line IS the
        # margin); past the canthus (wing) the bound skin triangle's deltas
        from juno_sculpt import lid_margins
        up_ids, _ = lid_margins(base, side)
        up_ids = up_ids[np.linalg.norm(VA[up_ids] - c, axis=1) > 1.04 * R]
        sg = 1.0 if side == "l" else -1.0
        xm = sg * (VA[up_ids, 0] - c[0])
        om = np.argsort(xm)
        up_ids, xm = up_ids[om], xm[om]
        xb = sg * (P0[:, 0, 0] - c[0])
        on_lid = (1 - np.clip((s - 0.97) / 0.12, 0, 1))[:, None]
        for k, Dk in S.items():
            dv = (Dk[tri] * bc[:, :, None]).sum(1).reshape(nu, nv, 3)
            dm = np.stack([np.interp(xb, xm, Dk[up_ids, q]) for q in range(3)], 1)   # along the margin (by lateral x)
            b0 = on_lid * dm + (1 - on_lid) * dv[:, 0]
            dv = (1 - rigid) * dv + rigid * b0[:, None]
            if k.startswith("eyeBlink") and p.get("wing_blink_collapse", 0) > 0 and np.abs(dm).max() > 0.3 * ri:
                # (only the blink that closes THIS side's lid; the other side's shape leaves the strip alone)
                # asset-eyes3: a closing lid folds the wing back toward the canthus (it stood up as a black
                # "tick" at the outer corner of closed eyes, judge r2): wing vertices pull toward the canthus
                # bottom vertex, more toward the tip
                P3 = Pp.reshape(nu, nv, 3)
                ic = int(np.searchsorted(s, 1.0))
                ic = min(max(ic, 1), nu - 1)
                canth = P3[ic - 1, 0]
                wf = np.clip((s - s[ic - 1]) / max(s.max() - s[ic - 1], 1e-9), 0, 1)[:, None, None]
                dv = dv - p["wing_blink_collapse"] * wf * (P3 - canth[None, None])
                # ... and the tall outer-corner wedge flattens onto its lash line (else a vertical hook stays, e2 blink1)
                ss_ = np.clip((s - 0.72) / 0.28, 0, 1)
                hc = (ss_ * ss_ * (3 - 2 * ss_))[:, None, None] * (1 - wf)
                dv = dv - p.get("corner_blink_flatten", 0.0) * hc * (P3 - P3[:, :1])
            dv = dv.reshape(-1, 3)
            sel = np.where(np.linalg.norm(dv, axis=1) > 1e-7)[0]
            if len(sel):
                shapes[k] = (sel, dv[sel])
        UV = np.stack(np.meshgrid(np.linspace(0, 1, nu), v, indexing="ij"), -1).reshape(-1, 2)
        UV[:, 0] = np.repeat(s / s.max(), nv)
        tris = []
        for i in range(nu - 1):
            for j in range(nv - 1):
                a_, b_ = i * nv + j, i * nv + j + 1
                c_, d_ = (i + 1) * nv + j, (i + 1) * nv + j + 1
                tris += [[a_, c_, b_], [b_, c_, d_]]
        tris = np.array(tris, np.int64)
        fn = np.cross(Pp[tris[:, 1]] - Pp[tris[:, 0]], Pp[tris[:, 2]] - Pp[tris[:, 0]])
        if (fn * (Pp[tris[:, 0]] - c)).sum(1).mean() < 0:
            tris = tris[:, [0, 2, 1]]
        n = len(Pp)
        hb = base.bidx["head"]
        parts.append(Part(f"lashline_{side}", "lashline", Pp, UV, tris, np.tile([hb, 0, 0, 0], (n, 1)),
                          np.tile([1.0, 0, 0, 0], (n, 1)), shapes=shapes, double_sided=True))
    # texture: u along (0 medial .. 1 tip), v across (0 = margin edge .. 1 = top edge). asset-eyes4: a soft warm
    # brown root shadow, gaussian across (peak just above the margin, soft both edges), faded at both ends, no ink core.
    nu_t, nv_t = 512, 64
    U, Vv = np.meshgrid((np.arange(nu_t) + 0.5) / nu_t, (np.arange(nv_t) + 0.5) / nv_t, indexing="xy")
    amax = LASHLINE.get("alpha_max", 0.5) if prm is None else dict(LASHLINE, **prm)["alpha_max"]
    a = amax * np.exp(-0.5 * ((Vv - 0.3) / 0.26) ** 2)
    a = a * np.clip((1.0 - U) / 0.12, 0, 1) * np.clip(U / 0.15, 0, 1)
    col = np.zeros((nv_t, nu_t, 4))
    col[..., 0], col[..., 1], col[..., 2] = 52, 32, 24
    col[..., 3] = np.clip(a, 0, 1) * 255
    im = Image.fromarray(np.clip(col, 0, 255).astype(np.uint8))
    im.save(os.path.join(texdir, "lashline_albedo.png"))
    im.resize((nu_t // 2, nv_t // 2), Image.LANCZOS).save(os.path.join(texdir, "lashline_albedo_1k.png"))
    mats["lashline"] = {"type": "lashes", "albedo": "tex/lashline_albedo.png",
                        "params": {"role": "upper lash line + wing (strip, alpha in albedo.a)", "roughness": 0.45}}
    return parts, mats
