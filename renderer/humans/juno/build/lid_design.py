"""asset-eyes2 — parametric lid redesign ON TOP of SCULPT's design fit (called from juno_design.stylize right after
the design_fit hook).

redesign(base, V, design) -> V   (stylize space, metres; returns a new array)

Concept eye (scratch/asset-concept/sheet_v4_0.png): an ALMOND aperture — outer canthus higher than the inner one, a
flatter upper arc that hoods the top ~18 % of the iris, a lower lid that just meets the iris bottom. The fitted head's
lid margins (juno_sculpt.lid_margins, topology-fixed) are rotated about the eyeball centre's x axis (so every vertex keeps
its distance to the ball centre: the margin stays ON its shell over the eye, the conjunctival tuck stays behind it); the
surrounding lid skin follows with a smooth falloff to the brow / cheek. Eye centre + radius are read from
base.design_fit (SCULPT's fitted ball) or the MH helper sphere — never hard-coded — so later global transforms (the
ASSET2 post-fit head scale) carry through. All amounts are in units of the fitted iris radius, so they scale with it.

DESIGN keys (juno_design.DESIGN may override; defaults in LID):
  lid_redesign   bool    master switch (JUNO_LID_REDESIGN=0 disables)
  lid_hood       float   upper margin centre height = iris_r * (1 - 2*lid_hood)   (fraction of the iris diameter hidden)
  lid_lower      float   lower margin centre height = -iris_r * (1 - 2*lid_lower) (fraction of the iris hidden by it)
  lid_outer_lift float   outer canthus raised by this * iris_r (inner canthus fixed)
  lid_peak       float   0..1 position (medial -> lateral) of the upper arc's highest point (concept: a touch medial)
  lid_crease     float   soft crease above the upper lid: inward push of this * iris_r on a band ~ lid_crease_off up
  lid_relax      int     Taubin iterations on the hidden inner-lid skin (margins pinned) to un-flip its triangles
"""
from __future__ import annotations

import os

import numpy as np

LID = {"lid_redesign": True, "lid_hood": 0.10, "lid_lower": 0.0, "lid_outer_lift": 0.25, "lid_peak": 0.42,
       "lid_crease": 0.05, "lid_crease_off_deg": 11.0, "lid_relax": 0,
       "lid_under_smooth": 10}   # asset-eyes3: Taubin iterations on the under-eye skin (no double crescent)


def _ss(a, b, x):
    t = np.clip((x - a) / (b - a), 0.0, 1.0)
    return t * t * (3 - 2 * t)


def eye_sphere(base, V, side):
    df = getattr(base, "design_fit", None)
    if df:
        return np.asarray(df[f"eye_center_{side}"], float), float(df["eye_radius"]), float(df.get("iris_radius", 0.0))
    ev = base.obj.group_verts(f"helper-{side}-eye")
    c = V[ev].mean(0)
    R = float(np.linalg.norm(V[ev] - c, axis=1).mean())
    return c, R, 0.47 * R


def _rot_x(q, ang):
    ca, sa = np.cos(ang), np.sin(ang)
    y = q[:, 1] * ca + q[:, 2] * sa
    z = -q[:, 1] * sa + q[:, 2] * ca
    return np.stack([q[:, 0], y, z], 1)


def _profile(az, el, azb):
    o = np.argsort(az)
    return np.interp(azb, az[o], el[o])


def redesign(base, V, design=None):
    d = dict(LID)
    if design:
        d.update({k: v for k, v in design.items() if k.startswith("lid_")})
    if not d["lid_redesign"] or os.environ.get("JUNO_LID_REDESIGN", "1") == "0":
        return V
    from juno_sculpt import lid_margins
    V = np.array(V, float, copy=True)
    skin = np.zeros(len(V), bool)
    skin[base.obj.group_verts("body")] = True
    rep = {}
    for side in ("l", "r"):
        c, R, ri = eye_sphere(base, V, side)
        if ri <= 0:
            ri = 0.47 * R
        sx = 1.0 if side == "l" else -1.0
        q = V - c
        r = np.linalg.norm(q, axis=1)
        az = np.arctan2(sx * q[:, 0], q[:, 2])          # + = lateral
        el = np.arctan2(q[:, 1], q[:, 2])                # + = up (rotation angle about x)
        rho = np.hypot(q[:, 1], q[:, 2])                 # distance to the x axis through the centre
        up, lo = lid_margins(base, side)
        upf, lof = up[r[up] > 1.04 * R], lo[r[lo] > 1.04 * R]   # front margin rows (the conjunctival tuck excluded)
        a0 = min(az[upf].min(), az[lof].min())
        a1 = max(az[upf].max(), az[lof].max())
        azb = np.linspace(a0, a1, 61)
        pu, pl = _profile(az[upf], el[upf], azb), _profile(az[lof], el[lof], azb)
        rhu, rhl = _profile(az[upf], rho[upf], azb), _profile(az[lof], rho[lof], azb)
        s = (azb - a0) / (a1 - a0)
        # ---- target margin heights (y about the centre) -> angle deltas (rotation about x, + = up)
        yu = rhu * np.sin(pu)
        yl = rhl * np.sin(pl)
        ic = np.argmin(np.abs(azb - 0.0))                # gaze axis column
        lift = d["lid_outer_lift"] * ri * _ss(0.15, 1.0, s)
        pk = d["lid_peak"]
        # hood shape: zero at the canthi, max a bit lateral of the peak so the arc flattens toward the outer corner
        bump_u = np.sin(np.pi * np.clip(s, 0, 1)) ** 0.55
        bump_u = bump_u * (1 + 0.35 * (s - pk))
        bump_l = np.sin(np.pi * np.clip(s, 0, 1)) ** 0.8
        dyu_c = ri * (1 - 2 * d["lid_hood"]) - yu[ic]    # (negative = down)
        dyl_c = -ri * (1 - 2 * d["lid_lower"]) - yl[ic]  # (positive = up)
        dyu = dyu_c * bump_u / max(bump_u[ic], 1e-6) + lift
        dyl = dyl_c * bump_l / max(bump_l[ic], 1e-6) + 0.85 * lift
        dau = dyu / np.maximum(rhu * np.cos(pu), 1e-6)    # dy = rho*cos(el)*d_el
        dal = dyl / np.maximum(rhl * np.cos(pl), 1e-6)
        # canthi: both lids meet -> the same angle at the ends
        for k in (0, -1):
            m = 0.5 * (dau[k] + dal[k])
            dau[k] = dal[k] = m
        # ---- field over the lid skin
        mid = 0.5 * (pu + pl)
        azc = np.clip(az, a0, a1)
        phu, phl, phm = np.interp(azc, azb, pu), np.interp(azc, azb, pl), np.interp(azc, azb, mid)
        Au, Al = np.interp(azc, azb, dau), np.interp(azc, azb, dal)
        span = a1 - a0
        lat_out = az > a1
        # beyond the outer canthus only the lift continues (fading over ~25 % of the span); beyond the inner one: nothing
        fade_az = (_ss(a0 - 0.12 * span, a0 + 0.02 * span, az) * (1 - _ss(a1 + 0.05 * span, a1 + 0.35 * span, az)))
        region = skin & (r < 2.4 * R) & (q[:, 2] > -0.45 * R)
        above = el >= phm
        t_up = np.degrees(el - phu)
        t_lo = np.degrees(phl - el)
        w_up = (1 - _ss(2.0, 34.0, t_up)) * (1 - _ss(1.45 * R, 2.05 * R, r))
        w_lo = (1 - _ss(1.0, 24.0, t_lo)) * (1 - _ss(1.35 * R, 1.9 * R, r))
        ang = np.where(above, w_up * Au, w_lo * Al) * fade_az * region
        # lift beyond the outer canthus: blend the two lids' end value
        ang = np.where(lat_out & region, np.where(above, w_up, w_lo) * dau[-1] * fade_az, ang)
        # smooth the angle field over the skin graph a little (no shear at the mid line / canthi)
        ang = _smooth_field(base, ang, region, iters=6, pin=np.concatenate([upf, lof]))
        newq = _rot_x(q, ang)
        # soft crease: a radial inward band above the upper margin (fades at the canthi)
        cr = d["lid_crease"] * ri
        if cr > 0:
            co = d["lid_crease_off_deg"]
            band = np.exp(-((t_up - co) / 4.0) ** 2) * above * region * (1 - _ss(1.3 * R, 1.8 * R, r))
            band *= np.interp(azc, azb, np.sin(np.pi * np.clip(s, 0, 1)) ** 0.7) * (az <= a1)
            nr = np.linalg.norm(newq, axis=1)
            newq = newq * (1 - cr * band / np.maximum(nr, 1e-9))[:, None]
        moved = np.abs(ang) > 1e-6
        V[moved] = c + newq[moved]
        if d.get("lid_under_smooth", 0):
            rep[f"under_{side}"] = _smooth_under(base, V, c, R, sx, lof, a0, a1, region, d["lid_under_smooth"])
        rep[side] = {"iris_r_mm": round(ri * 1e3, 2), "up_c_mm": round(float(yu[ic]) * 1e3, 2),
                     "lo_c_mm": round(float(yl[ic]) * 1e3, 2), "dy_up_c_mm": round(float(dyu_c) * 1e3, 2),
                     "dy_lo_c_mm": round(float(dyl_c) * 1e3, 2), "lift_mm": round(float(lift[-1]) * 1e3, 2),
                     "moved": int(moved.sum())}
    if d["lid_relax"]:
        rep["relax"] = _relax_inner(base, V, d["lid_relax"])
    base.lid_design = rep
    return V


def _adjacency(base):
    import scipy.sparse as sp
    t = base.tris if hasattr(base, "tris") else None
    if t is None:
        from juno_base import tri_of
        t = np.array([tt for f in base.obj.faces for tt in tri_of([vi for vi, _ in f])], np.int64)
    n = base.nV
    e = np.concatenate([t[:, [0, 1]], t[:, [1, 2]], t[:, [2, 0]]])
    A = sp.coo_matrix((np.ones(len(e) * 2), (np.r_[e[:, 0], e[:, 1]], np.r_[e[:, 1], e[:, 0]])), shape=(n, n)).tocsr()
    A.data[:] = 1.0
    deg = np.maximum(np.asarray(A.sum(1)).ravel(), 1)
    return A, deg, t


def _smooth_field(base, f, region, iters, pin):
    A, deg, _ = _adjacency(base)
    f = f.copy()
    keep = f[pin].copy()
    for _ in range(iters):
        f = np.where(region, 0.5 * f + 0.5 * (A @ f) / deg, f)
        f[pin] = keep
    return f


def _smooth_under(base, V, c, R, sx, lof, a0, a1, region, iters):
    """asset-eyes3: Taubin smoothing of the lower-lid / under-eye skin (the MH lower-lid fold loops read as a
    double dark crescent, "eye bags", in every light). Region: below the lower margin (from ~2 deg under it), within
    ~1.9 R of the ball, front half; weight ramps in under the margin (the margin itself and its first ring pinned, so
    the lid line and the tearline/occlusion strips do not move) and fades out toward the cheek. In place."""
    A, deg, _ = _adjacency(base)
    q = V - c
    r = np.linalg.norm(q, axis=1)
    az = np.arctan2(sx * q[:, 0], q[:, 2])
    el = np.arctan2(q[:, 1], q[:, 2])
    pl = _profile(az[lof], el[lof], np.clip(az, a0, a1))
    t_lo = np.degrees(pl - el)
    w = _ss(2.0, 7.0, t_lo) * (1 - _ss(1.55 * R, 1.95 * R, r)) * (q[:, 2] > 0.1 * R) * region
    w *= _ss(a0 - 0.1, a0 + 0.15, az) * (1 - _ss(a1 + 0.05, a1 + 0.35, az))
    w[lof] = 0
    P = V.copy()
    for _ in range(iters):
        for lam in (0.55, -0.58):
            L = (A @ P) / deg[:, None] - P
            P = P + lam * w[:, None] * L
    dv = np.linalg.norm(P - V, axis=1)
    V[:] = P
    return {"verts": int((w > 0.01).sum()), "max_move_mm": round(float(dv.max()) * 1e3, 3)}


def _flips(V, t, N0, sel):
    n = np.cross(V[t[:, 1]] - V[t[:, 0]], V[t[:, 2]] - V[t[:, 0]])
    return int(((n * N0).sum(1)[sel] < 0).sum())


def _relax_inner(base, V, iters):
    """Taubin on the hidden inner-lid skin (between the margin and the ball; margins pinned), each vertex projected back
    onto its own distance from the eye centre so the tuck stays behind the margin. Returns flip counts (vs the MH
    base orientation) on the inner-lid triangles before/after."""
    from juno_sculpt import lid_margins
    A, deg, t = _adjacency(base)
    V0 = base.obj.v
    N0 = np.cross(V0[t[:, 1]] - V0[t[:, 0]], V0[t[:, 2]] - V0[t[:, 0]])
    skin = np.zeros(len(V), bool)
    skin[base.obj.group_verts("body")] = True
    out = {}
    for side in ("l", "r"):
        c, R, _ = eye_sphere(base, V, side)
        q = V - c
        r = np.linalg.norm(q, axis=1)
        up, lo = lid_margins(base, side)
        pin = np.zeros(len(V), bool)
        pin[up] = pin[lo] = True
        inner = skin & (r < 1.06 * R) & (r > 0.6 * R) & (q[:, 2] > 0.0) & ~pin
        sel = np.isin(t, np.where(inner)[0]).any(1)
        before = _flips(V, t, N0, sel)
        rad = r.copy()
        W = inner.astype(float)
        P = V.copy()
        for _ in range(iters):
            for lam in (0.5, -0.53):
                L = (A @ P) / deg[:, None] - P
                P = P + lam * W[:, None] * L
        qq = P[inner] - c
        P[inner] = c + qq / np.linalg.norm(qq, axis=1, keepdims=True) * rad[inner, None]
        after = _flips(P, t, N0, sel)
        if after <= before:
            V[inner] = P[inner]
        out[side] = {"inner_verts": int(inner.sum()), "flips_before": before, "flips_after": after,
                     "applied": bool(after <= before)}
    return out
