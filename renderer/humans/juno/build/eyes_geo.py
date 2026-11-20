"""asset — eye geometry for eyes_mouth.py: lid-aperture fit (ray cast against the skin), eyeball with a concave
iris dish, cornea shell, eye-occlusion shell and tearline strips that follow every lid shape.

Everything is measured from the Base (eye bone centre + the skin around it), never hard-coded, so it refits when the
lead changes the design. Eye local frame = asset axes (x = her left, y up, z = gaze at rest).
"""
from __future__ import annotations

import numpy as np

from juno_base import smoothstep
from parts import Part

D2R = np.pi / 180
IRIS_W = 0.64      # iris diameter / visible eye width ( asset-eyes2: concept-big iris)
PUPIL_K = 0.45     # pupil / iris diameter ( asset-eyes2: bigger, concept; eyes4 .43 -> .45 warmth)
IRIS_GROW = 1.07   # asset-eyes4: iris radius x this after the fit (judge r4: warmer, bigger iris)
IRIS_W_MAX = 0.72  # cap when the aperture is too tall for the cover rule
TOP_COVER = 0.12   # upper lid covers this fraction of the iris diameter at rest
GAZE_DEG = 12.0    # cornea must clear the rest lids with this much gaze turn


def ray_cast(o, dirs, P, tris, tmin=0.0, tmax=1e9, chunk=2048, far=False):
    """Nearest (far=True: farthest) hit of rays o + t*dirs (t in (tmin,tmax)) on triangles.
    -> t (inf = miss), tri idx, bary (n,3)."""
    a, b, c = P[tris[:, 0]], P[tris[:, 1]], P[tris[:, 2]]
    e1, e2 = b - a, c - a
    n = len(dirs)
    T = np.full(n, np.inf)
    TI = np.full(n, -1, np.int64)
    BU = np.zeros(n)
    BV = np.zeros(n)
    for s in range(0, n, chunk):
        d = dirs[s:s + chunk][:, None, :]
        pv = np.cross(d, e2[None])
        det = (e1[None] * pv).sum(-1)
        ok = np.abs(det) > 1e-14
        inv = np.where(ok, 1.0 / np.where(ok, det, 1), 0)
        tv = (o - a)[None]
        u = (tv * pv).sum(-1) * inv
        qv = np.cross(tv, e1[None])
        v = (d * qv).sum(-1) * inv
        t = (e2[None] * qv).sum(-1) * inv
        hit = ok & (u >= -1e-9) & (v >= -1e-9) & (u + v <= 1 + 1e-9) & (t > tmin) & (t < tmax)
        t = np.where(hit, t, np.inf)
        j = np.argmax(np.where(hit, t, -np.inf), 1) if far else np.argmin(t, 1)
        r = np.arange(len(j))
        T[s:s + chunk] = t[r, j]
        TI[s:s + chunk] = j
        BU[s:s + chunk] = u[r, j]
        BV[s:s + chunk] = v[r, j]
    bary = np.stack([1 - BU - BV, BU, BV], 1)
    return T, TI, bary


def polar_dir(theta, phi):
    return np.stack([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)], -1)


def eye_uv(theta, phi):
    r = 0.5 * theta / np.pi
    return np.stack([0.5 + r * np.cos(phi), 0.5 + r * np.sin(phi)], -1)


def polar_grid(thetas, nphi, pos_fn, weld_base=0, pole0=True, pole1=False):
    """Grid of rings (thetas) x (nphi+1) columns (last column duplicates the first, welded). pos_fn(theta, phi)->(...,3).
    pole0/pole1: first/last ring is a pole (welded to one point, degenerate tris dropped)."""
    nt = len(thetas)
    TH, PH = np.meshgrid(thetas, np.linspace(0, 2 * np.pi, nphi + 1), indexing="ij")
    P = pos_fn(TH, PH).reshape(-1, 3)
    UV = eye_uv(TH, PH).reshape(-1, 2)
    W = nphi + 1
    weld = np.arange(nt * W).reshape(nt, W)
    weld[:, nphi] = weld[:, 0]
    if pole0:
        weld[0, :] = weld[0, 0]
    if pole1:
        weld[-1, :] = weld[-1, 0]
    tris = []
    for i in range(nt - 1):
        for j in range(nphi):
            a, b, c, d = i * W + j, i * W + j + 1, (i + 1) * W + j, (i + 1) * W + j + 1
            # outward-facing CCW (seen from outside, theta grows outward from the +z pole, phi CCW about +z)
            if not (pole0 and i == 0):
                tris.append([a, c, b])
            if not (pole1 and i == nt - 2):
                tris.append([b, c, d])
    return P, UV, np.array(tris, np.int64), weld.reshape(-1) + weld_base, TH.reshape(-1), PH.reshape(-1)


class EyeFit:
    """Lid-aperture fit for one eye."""

    def __init__(self, base, S, side, nphi=96):
        self.side = side
        L = "Left" if side == "l" else "Right"
        c, r0 = base.eye[side]
        self.c, self.r0 = np.asarray(c, float), float(r0)
        VA = base.VA
        near = base.skin_mask & (np.linalg.norm(VA - c, axis=1) < 2.2 * r0)
        tm = near[base.tris].all(1)
        self.tris = base.tris[tm]
        self.base = base
        # lid-moving shapes: anything in S that moves skin near this eye
        self.lid_shapes = [k for k, d in S.items() if np.abs(d[near]).max() > 2e-5]
        self.S = S
        # eyeball = the sculpt's ball (base.eye radius; the lid skin is wrapped onto 1.035*R). Report how close any
        # lid state comes (face_shapes' blink must slide on that sphere; < 1.0 = the lid cuts into the ball).
        from juno_base import vertex_normals
        nv = np.where(near)[0]
        self.lid_clear = {}
        for st in [None] + [(f"eyeBlink{L}", w) for w in (0.5, 1.0)] + [(f"eyeSquint{L}", 1.0), (f"eyeLookDown{L}", 1.0)]:
            if st is not None and st[0] not in S:
                continue
            P = VA if st is None else VA + st[1] * S[st[0]]
            d = P[nv] - c
            dn = np.linalg.norm(d, axis=1)
            sel = d[:, 2] / dn > np.cos(70 * D2R)
            self.lid_clear["rest" if st is None else f"{st[0]}@{st[1]}"] = round(float(dn[sel].min() / r0), 3)
        self.R = self.r0
        self.outer_min = self.lid_clear["rest"] * self.r0
        # aperture margin per phi: smallest theta whose ray hits skin beyond ~R (the lid crosses the eyeball there)
        self.phi = np.linspace(0, 2 * np.pi, nphi, endpoint=False)
        thetas = np.arange(1.0, 85.0, 0.25) * D2R
        TH, PH = np.meshgrid(thetas, self.phi, indexing="ij")
        dirs = polar_dir(TH, PH).reshape(-1, 3)
        t, ti, bary = ray_cast(self.c, dirs, VA, self.tris, tmin=0.97 * self.R, tmax=1.8 * self.R)
        hit = np.isfinite(t).reshape(len(thetas), nphi)
        first = np.argmax(hit, 0)
        assert hit.any(0).all(), "eye aperture not closed by skin in some direction"
        self.theta_m = thetas[first]
        k = first * nphi + np.arange(nphi)
        self.m_tri, self.m_bary = ti[k], bary[k]
        self.m_pos = self.c + dirs[k] * t[k][:, None]
        up = self.theta_m[np.argmin(np.abs(self.phi - np.pi / 2))]
        dn_ = self.theta_m[np.argmin(np.abs(self.phi - 1.5 * np.pi))]
        self.theta_up, self.theta_down = float(up), float(dn_)
        self.phi_medial = np.pi if side == "l" else 0.0
        # iris: animated-film big — diameter ~ IRIS_W of the visible (front-projected) aperture width, so the upper lid
        # overlaps it at rest; pupil = PUPIL_K of the iris
        ap = self.R * np.sin(self.theta_m)[:, None] * np.stack([np.cos(self.phi), np.sin(self.phi)], 1)
        self.vis_w = float(ap[:, 0].max() - ap[:, 0].min())
        self.vis_h = float(ap[:, 1].max() - ap[:, 1].min())
        r_cover = self.R * np.sin(up) / (1 - 2 * TOP_COVER)          # iris radius for TOP_COVER lid overlap
        ri = np.clip(max(IRIS_W * self.vis_w / 2, r_cover), 0.45 * self.R, min(IRIS_W_MAX * self.vis_w / 2, 0.8 * self.R))
        ri = float(min(ri * IRIS_GROW, 0.8 * self.R))
        # where the upper lid margin would have to sit (above the ball centre) for the cover with this iris
        self.lid_up_want = float(ri * (1 - 2 * TOP_COVER))
        self.theta_iris = float(np.arcsin(ri / self.R))
        self.theta_limbus = self.theta_iris * 1.04
        self.iris_top_cover = float((ri - self.R * np.sin(up)) / (2 * ri))
        # cornea bulge: stays under the rest lids even with the gaze turned GAZE_DEG toward them
        tc = np.clip(self.theta_m - GAZE_DEG * D2R, 0, None)
        clear = (self.lid_clear["rest"] - 1.0 - 0.012) * self.R
        prof = self.bulge_profile(tc)
        self.bulge = float(np.clip(clear / max(prof.max(), 1e-3), 0.02 * self.R, 0.07 * self.R))
        self.eps_cornea = 0.004 * self.R

    def bulge_profile(self, theta):
        s = np.clip(theta / self.theta_limbus, 0, 1)
        return (1 - s * s) ** 2

    def cornea_radius(self, theta):
        return self.R + self.eps_cornea + self.bulge * self.bulge_profile(theta)

    def margin_deltas(self, idx=None):
        """Per shape: (nphi,3) displacement of the lid margin points (barycentric copy of the skin deltas)."""
        tri = self.tris[self.m_tri]
        out = {}
        for k in self.lid_shapes:
            d = (self.S[k][tri] * self.m_bary[:, :, None]).sum(1)
            if np.abs(d).max() > 1e-6:
                out[k] = d
        return out

    def report(self):
        return dict(side=self.side, R_mm=round(self.R * 1000, 2), lid_clear_over_R=self.lid_clear,
                    vis_w_mm=round(self.vis_w * 1000, 1), vis_h_mm=round(self.vis_h * 1000, 1),
                    iris_top_cover=round(self.iris_top_cover, 3),
                    lid_up_now_mm=round(self.R * np.sin(self.theta_up) * 1000, 1), lid_up_want_mm=round(self.lid_up_want * 1000, 1), theta_up=round(self.theta_up / D2R, 1),
                    theta_down=round(self.theta_down / D2R, 1), theta_iris=round(self.theta_iris / D2R, 1),
                    bulge_mm=round(self.bulge * 1000, 2), lid_shapes=len(self.lid_shapes))


def _bone_part(name, mat, P, UV, tris, weld, bone, blend="opaque", shapes=None):
    n = len(P)
    return Part(name, mat, P, UV, tris, np.tile([bone, 0, 0, 0], (n, 1)), np.tile([1.0, 0, 0, 0], (n, 1)), weld=weld,
                blend=blend, shapes=shapes)


def eyeball_part(fit, side, bone, nphi=64, dish=0.07):
    """Sclera sphere + concave iris dish, hard edge at the limbus (separate weld ids)."""
    R, ti = fit.R, fit.theta_iris
    rho_i = R * np.sin(ti)
    zi = R * np.cos(ti)

    def iris_pos(th, ph):
        s = th / ti
        rho = rho_i * s
        z = zi - dish * rho_i * (1 - s * s)
        return fit.c + np.stack([rho * np.cos(ph), rho * np.sin(ph), z], -1)

    def scl_pos(th, ph):
        return fit.c + R * polar_dir(th, ph)

    t_iris = ti * np.sqrt(np.linspace(0, 1, 13))  # denser toward the limbus
    t_iris[0] = 0.0
    t_scl = np.concatenate([ti + (np.pi * 0.62 - ti) * np.linspace(0, 1, 16) ** 1.4,
                            np.linspace(np.pi * 0.62, np.pi, 9)[1:]])
    P1, UV1, T1, W1, _, _ = polar_grid(t_iris, nphi, iris_pos, 0, pole0=True)
    P2, UV2, T2, W2, _, _ = polar_grid(t_scl, nphi, scl_pos, len(P1), pole0=False, pole1=True)
    P = np.concatenate([P1, P2])
    UV = np.concatenate([UV1, UV2])
    T = np.concatenate([T1, T2 + len(P1)])
    Wd = np.concatenate([W1, W2])
    return _bone_part(f"eye_{side}", f"eye_{side}", P, UV, T, Wd, bone)


def cornea_part(fit, side, bone, nphi=64):
    tc = np.concatenate([fit.theta_limbus * np.sqrt(np.linspace(0, 1, 12)), np.linspace(fit.theta_limbus, 88 * D2R, 9)[1:]])
    tc[0] = 0

    def pos(th, ph):
        return fit.c + fit.cornea_radius(th)[..., None] * polar_dir(th, ph)

    P, UV, T, Wd, _, _ = polar_grid(tc, nphi, pos, 0, pole0=True)
    return _bone_part(f"cornea_{side}", "cornea", P, UV, T, Wd, bone, blend="alpha")


def _shell_radius(fit, theta, extra):
    return fit.cornea_radius(theta) + extra


def lid_strip(fit, name, mat, bone_head, phis, th_out, th_in, extra, ncross=6):
    """Strip on the eye surface along the lid margin. phis: (M,) angles (sorted, may wrap); th_out/th_in: (M,) thetas of
    the outer (under the lid) and inner edge. Shapes = barycentric copy of the margin skin deltas, re-projected onto the
    shell surface (so the strip slides over the eye with the lid). UV: u along the strip 0..1, v across (0 = lid side)."""
    M = len(phis)
    md = fit.margin_deltas()
    # margin deltas at arbitrary phi: periodic interpolation of the nphi samples
    ph0 = fit.phi

    def interp(vals, ph):
        x = np.concatenate([ph0 - 2 * np.pi, ph0, ph0 + 2 * np.pi])
        v = np.concatenate([vals, vals, vals])
        return np.interp(np.mod(ph, 2 * np.pi), x, v)

    vs = np.linspace(0, 1, ncross)
    P, UV, TH, PH, V = [], [], [], [], []
    for i in range(M):
        for j, v in enumerate(vs):
            th = th_out[i] + (th_in[i] - th_out[i]) * v
            d = polar_dir(np.array(th), np.array(phis[i]))
            P.append(fit.c + _shell_radius(fit, th, extra) * d)
            UV.append([i / max(M - 1, 1), v])
            TH.append(th); PH.append(phis[i]); V.append(v)
    P, UV = np.array(P), np.array(UV)
    tris = []
    for i in range(M - 1):
        for j in range(ncross - 1):
            a, b = i * ncross + j, i * ncross + j + 1
            cc, dd = (i + 1) * ncross + j, (i + 1) * ncross + j + 1
            tris += [[a, cc, b], [b, cc, dd]]
    tris = np.array(tris, np.int64)
    # orient outward (+radial)
    fn = np.cross(P[tris[:, 1]] - P[tris[:, 0]], P[tris[:, 2]] - P[tris[:, 0]])
    if (fn * (P[tris[:, 0]] - fit.c)).sum(1).mean() < 0:
        tris = tris[:, [0, 2, 1]]
    shapes = {}
    PHa = np.array(PH)
    for k, dm in md.items():
        dv = np.stack([interp(dm[:, q], PHa) for q in range(3)], 1)
        Q = P + dv
        # re-project onto the shell surface along the radial direction
        rel = Q - fit.c
        rn = np.linalg.norm(rel, axis=1)
        u = rel / rn[:, None]
        th = np.arccos(np.clip(u[:, 2], -1, 1))
        Q = fit.c + u * _shell_radius(fit, th, extra)[:, None]
        d = Q - P
        sel = np.where(np.linalg.norm(d, axis=1) > 1e-7)[0]
        if len(sel):
            shapes[k] = (sel, d[sel])
    n = len(P)
    return Part(name, mat, P, UV, tris, np.tile([bone_head, 0, 0, 0], (n, 1)), np.tile([1.0, 0, 0, 0], (n, 1)),
                blend="alpha", shapes=shapes)


def occlusion_part(fit, side, bone_head):
    M = 97
    phis = np.linspace(0, 2 * np.pi, M)
    tm = np.interp(phis, np.concatenate([fit.phi, [2 * np.pi]]), np.concatenate([fit.theta_m, fit.theta_m[:1]]))
    th_out = tm + 4 * D2R                 # tucked under the lid
    th_in = tm - np.clip(0.22 * tm, 4 * D2R, 8 * D2R)
    return lid_strip(fit, f"eye_occl_{side}", "eye_occlusion", bone_head, phis, th_out, th_in, 0.00022, ncross=7)


def tearline_part(fit, side, bone_head):
    """Lower lid margin from the lateral corner round to just past the medial corner (caruncle)."""
    med = fit.phi_medial
    lat = med + np.pi
    # lower lid: phi from lateral corner, through 270 deg (down), to the medial corner, then 25 deg up past it
    if side == "l":   # medial = pi, lateral = 0 ; going down = decreasing phi from 2pi -> pi
        phis = np.linspace(2 * np.pi + 8 * D2R, np.pi - 28 * D2R, 80)
    else:             # medial = 0, lateral = pi ; going down: pi -> 2pi, then past 0
        phis = np.linspace(np.pi - 8 * D2R, 2 * np.pi + 28 * D2R, 80)
    del lat
    tm = np.interp(np.mod(phis, 2 * np.pi), np.concatenate([fit.phi, [2 * np.pi]]), np.concatenate([fit.theta_m, fit.theta_m[:1]]))
    th_out = tm + 1.6 * D2R
    th_in = tm - 2.4 * D2R
    return lid_strip(fit, f"tearline_{side}", "tearline", bone_head, phis, th_out, th_in, 0.00034, ncross=5)


def conform_lids(base, S, k_r=1.035, report=None):
    """IN PLACE: every shape in S that would pull front-facing skin around an eye inside the lid sphere (k_r * R, the
    sphere the sculpt wraps the lids on) is corrected so that skin slides OVER the ball instead (radial push to
    min(rest distance, k_r*R)). Fixes blink/squint lids vanishing behind the big sculpted eyeballs. Shapes are additive,
    so combos can still cut in slightly; single shapes at weight 1 are clean (and weight 0.5 nearly so)."""
    VA = base.VA
    out = {}
    for side in ("l", "r"):
        c, R = base.eye[side]
        q = VA - c
        d0 = np.linalg.norm(q, axis=1)
        cand = np.where(base.skin_mask & (d0 > 0.99 * R) & (d0 < 1.8 * R) & (q[:, 2] > np.cos(80 * D2R) * d0))[0]
        floor = np.minimum(d0[cand], k_r * R)
        for name, d in S.items():
            if not np.any(d[cand]):
                continue
            for w in (1.0,):
                p = VA[cand] + w * d[cand]
                rel = p - c
                dn = np.linalg.norm(rel, axis=1)
                bad = dn < floor - 1e-6
                if not bad.any():
                    continue
                fix = rel[bad] / dn[bad, None] * floor[bad, None] + c
                d[cand[bad]] = fix - VA[cand[bad]]
                out[f"{name}/{side}"] = int(bad.sum())
    if report is not None:
        report["conform_lids"] = out
    return out


CARUNCLE = {"theta_in_deg": 1.0,   # dome centre this far inside the medial margin (toward the iris)
            "a_vert": 0.06,        # half-height along the lid (x R)
            "a_in": 0.05,          # half-length toward the iris (x R)
            "h_out": 0.022,        # dome height above the eye shell (x R)
            "sink": 0.025}          # base pushed under the shell (x R) so no gap shows at its edge


def caruncle_part(fit, side, bone_head, n_u=20, n_v=10):
    """asset-eyes4: the CARUNCLE — a small wet pink mound filling the inner canthus, sitting on the eye shell just
    inside the medial lid margin (half under the corner skin). A squashed dome in the local (tangent) frame of the shell;
    rides the lids with the medial margin's own shape deltas (the corner barely moves in a blink, so it stays put)."""
    p = CARUNCLE
    R = fit.R
    i_med = int(np.argmin(np.abs(np.angle(np.exp(1j * (fit.phi - fit.phi_medial))))))
    th0 = float(fit.theta_m[i_med]) - p["theta_in_deg"] * D2R
    ph0 = fit.phi_medial
    n0 = polar_dir(np.array(th0), np.array(ph0))
    c0 = fit.c + fit.cornea_radius(np.array(th0)) * n0 - p["sink"] * R * n0
    # tangent frame: e_in = toward the eye's front pole (decreasing theta), e_v = along the lid (vertical-ish)
    e_in = polar_dir(np.array(th0 - 0.01), np.array(ph0)) - polar_dir(np.array(th0), np.array(ph0))
    e_in = e_in - n0 * (e_in @ n0)
    e_in /= np.linalg.norm(e_in)
    e_v = np.cross(n0, e_in)
    uu, vv = np.meshgrid(np.linspace(0, 2 * np.pi, n_u + 1), np.linspace(0, 0.5 * np.pi, n_v), indexing="ij")
    # dome: ring radius cos(v), height sin(v); slight teardrop (fuller on the medial side)
    rr = np.cos(vv)
    x_in = p["a_in"] * R * rr * np.cos(uu) * np.where(np.cos(uu) < 0, 1.25, 0.85)
    x_v = p["a_vert"] * R * rr * np.sin(uu)
    h = (p["h_out"] + p["sink"]) * R * np.sin(vv) ** 0.8
    P = c0 + x_in[..., None] * e_in + x_v[..., None] * e_v + h[..., None] * n0
    P = P.reshape(-1, 3)
    UV = np.stack([uu / (2 * np.pi), vv / (0.5 * np.pi)], -1).reshape(-1, 2)
    W = n_v
    tris = []
    for i in range(n_u):
        for j in range(n_v - 1):
            a, b, c, d = i * W + j, i * W + j + 1, (i + 1) * W + j, (i + 1) * W + j + 1
            tris += [[a, c, b], [b, c, d]]
    tris = np.array(tris, np.int64)
    fn = np.cross(P[tris[:, 1]] - P[tris[:, 0]], P[tris[:, 2]] - P[tris[:, 0]])
    cen = P[tris].mean(1)
    if (fn * (cen - (c0 - 0.05 * R * n0))).sum(1).mean() < 0:
        tris = tris[:, [0, 2, 1]]
    weld = np.arange(len(P)).reshape(n_u + 1, n_v)
    weld[n_u] = weld[0]
    weld[:, n_v - 1] = weld[0, n_v - 1]                  # apex
    shapes = {}
    for k, dm in fit.margin_deltas().items():
        d = dm[i_med]
        if np.linalg.norm(d) > 1e-7:
            shapes[k] = (np.arange(len(P)), np.tile(0.6 * d, (len(P), 1)))
    n = len(P)
    return Part(f"caruncle_{side}", "caruncle", P, UV, tris, np.tile([bone_head, 0, 0, 0], (n, 1)),
                np.tile([1.0, 0, 0, 0], (n, 1)), weld=weld.reshape(-1), shapes=shapes)
