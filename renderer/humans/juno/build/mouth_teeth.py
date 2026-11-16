"""asset-face r3 — procedural stylised teeth + gums, fitted to Juno's mouth (replaces the MH helper rows).

build(base, rig, cav_verts) -> [Part teeth_upper (head), teeth_lower (jaw), gums_upper (head), gums_lower (jaw)], REPORT

Landmark-driven, no coordinates:
  * the upper arch's labial curve is the inner lip lining (MH mouth-cavity vertices in the upper-lip band, per-x front-most,
    fitted z = a + b x^2 + c x^4) pulled back by CLEAR; it is only used for |x| <= the lining fit range, beyond that the
    curve continues on the fit (premolars/molar sit in the cheek corridor).
  * heights from the lip seam (rig.seam_y): upper incisal edge HANG below the seam (hidden behind the lower lip when the
    mouth is closed, shows ~3 mm when the upper lip lifts), a curve of Spee lifts the back teeth.
  * the lower arch = the upper arch offset inward (overjet) and its incisal edge OVERBITE above the upper edge (lower
    incisors sit behind the uppers); posteriors meet the uppers.
  * sizes in mouth half-widths (mw) so the lead's proportion changes carry through.
Each tooth is its own closed mesh (rounded-rectangle sections lofted from the incisal edge to the root, tapering at the neck,
thin at the incisal edge); gums = a scalloped bead along the necks.
"""
from __future__ import annotations

import numpy as np

from parts import Part

MW0 = 0.02582                     # mouth half-width the sizes below were tuned on (metres)
CLEAR = 0.0011                    # labial face of the upper teeth behind the lip lining
HANG = 0.0013                     # upper incisal edge below the lip seam at rest
SPEE = 0.0018                     # back teeth rise (curve of Spee) at the last tooth
OVERBITE = 0.0019                 # lower incisal edge above the upper incisal edge (hidden behind the uppers)
OVERJET = 0.0005                  # gap lower labial <- upper lingual at the incisal third

# per side from the midline: (name, width, crown height, thickness at the neck, thickness at the edge) in metres @ MW0
UPPER = [("c", 0.0080, 0.0098, 0.0060, 0.0016), ("l", 0.0064, 0.0084, 0.0055, 0.0016), ("k", 0.0072, 0.0092, 0.0070, 0.0026),
         ("p1", 0.0066, 0.0076, 0.0080, 0.0050), ("p2", 0.0064, 0.0070, 0.0080, 0.0055), ("m1", 0.0095, 0.0064, 0.0098, 0.0075)]
LOWER = [("c", 0.0052, 0.0082, 0.0052, 0.0014), ("l", 0.0057, 0.0084, 0.0055, 0.0015), ("k", 0.0066, 0.0084, 0.0066, 0.0024),
         ("p1", 0.0066, 0.0062, 0.0072, 0.0050), ("p2", 0.0068, 0.0058, 0.0078, 0.0055), ("m1", 0.0100, 0.0052, 0.0095, 0.0075)]
# asset-mouth4: stylized bigger, brighter upper row filling the wider mouth; lower row a touch smaller
UPPER_WK, UPPER_HK, LOWER_K = 1.06, 1.22, 0.94
UPPER = [(n, w * UPPER_WK, h * UPPER_HK, a, b) for n, w, h, a, b in UPPER]
LOWER = [(n, w * LOWER_K, h * LOWER_K, a, b) for n, w, h, a, b in LOWER]
GAP = 0.00012                     # interproximal gap (metres @ MW0) so each tooth reads (mouth4: was .18 mm)
ROOT = 0.0030                     # crown continues this far into the gum
NU = 12                           # section points per tooth

REPORT = {}


def _lining_fit(base, rig, cav_verts, sgn):
    """front-most cavity lining z(x) in the upper (sgn=+1) / lower (sgn=-1) lip band -> poly coeffs in x^2, x range."""
    VA, mw = base.VA, rig.mw
    P = VA[cav_verts]
    sy = np.interp(np.clip(P[:, 0], -mw, mw), rig.seam_x, rig.seam_y)
    dy = sgn * (P[:, 1] - sy)
    k = (dy > 0.06 * mw) & (dy < 0.55 * mw)
    Q = P[k]
    xs, zs = [], []
    for xb in np.arange(0, 1.45 * mw, 0.08 * mw):
        m = np.abs(np.abs(Q[:, 0]) - xb) < 0.06 * mw
        if m.sum() >= 2:
            xs.append(xb); zs.append(Q[m, 2].max())
    xs, zs = np.array(xs), np.array(zs)
    # the lining recedes smoothly toward the cheeks; a bin that drops > 4 mm below its predecessor caught the deep bag
    keep = np.ones(len(xs), bool)
    last = zs[0]
    for i in range(1, len(xs)):
        if zs[i] < last - 0.008:
            keep[i:] = False
            break
        last = zs[i]
    A = np.c_[np.ones_like(xs), xs ** 2, xs ** 4]
    co = np.linalg.lstsq(A[keep], zs[keep], rcond=None)[0]
    co[2] = min(co[2], 0.0)
    co[0] += min(0.0, float(np.clip((zs - A @ co)[keep].min(), -0.0015, 0)))
    return co, float(xs[keep].max()), xs[keep], zs[keep]


def arch_curves(base, rig, cav_verts):
    """labial z(x) of the upper / lower teeth arches (lining fits minus CLEAR) -> (fu, fl) callables."""
    co_u = _lining_fit(base, rig, cav_verts, +1)[0]
    co_l = _lining_fit(base, rig, cav_verts, -1)[0]
    fu = lambda x: co_u[0] + co_u[1] * x ** 2 + co_u[2] * x ** 4 - CLEAR
    th = UPPER[0][4] * rig.mw / MW0 + OVERJET * rig.mw / MW0 + 0.0004 * rig.mw / MW0
    fl = lambda x: np.minimum(fu(x) - th, co_l[0] + co_l[1] * x ** 2 + co_l[2] * x ** 4 - CLEAR)
    return fu, fl


def _curve(co, xmax, n=600):
    x = np.linspace(0, xmax, n)
    z = co[0] + co[1] * x ** 2 + co[2] * x ** 4
    s = np.r_[0, np.cumsum(np.hypot(np.diff(x), np.diff(z)))]
    return x, z, s


def _at(cur, s):
    x, z, ss = cur
    xi, zi = np.interp(s, ss, x), np.interp(s, ss, z)
    e = 1e-4
    t = np.array([np.interp(s + e, ss, x) - np.interp(s - e, ss, x), 0, np.interp(s + e, ss, z) - np.interp(s - e, ss, z)])
    t /= np.linalg.norm(t)
    nrm = np.array([-t[2], 0, t[0]])            # outward (labial/buccal) in xz for the +x half
    if nrm[2] < 0 and abs(nrm[2]) > abs(nrm[0]):
        nrm = -nrm
    if nrm[0] < -1e-6 and xi > 1e-4:
        nrm = -nrm
    return np.array([xi, 0, zi]), t, nrm


def _tooth(c, t, n, up, width, height, th_neck, th_edge, kind, tilt):
    """closed tooth mesh. c = labial-incisal centre point, t along arch, n labial, up = toward the root (+y upper, -y
    lower). Returns P (k,3), tris."""
    rings = []
    vs = np.r_[0.0, 0.05, 0.14, 0.32, 0.56, 0.80, 1.0]
    vs = np.r_[vs, 1.0 + ROOT / height]
    ang = np.linspace(0, 2 * np.pi, NU, endpoint=False)
    for k, v in enumerate(vs):
        vv = min(v, 1.0)
        inc = kind in ("c", "l")
        # width: rounded incisal corners, narrower neck
        wv = width * (1 - (0.30 if inc else 0.18) * vv ** 1.6)
        if v < 0.16:
            r = 1 - v / 0.16
            wv *= np.sqrt(max(1 - r ** 2 * (0.45 if inc else 0.45), 0.0))   # mouth4: squarer incisal corners (was .75/.55: a jagged comb at card size) * (1 if inc else 1) + (0.0)
        thv = th_edge + (th_neck - th_edge) * min(vv / 0.75, 1) ** 0.7
        if v < 0.03:
            thv *= 0.7
        if k == 0:
            wv, thv = wv * 0.45, thv * 0.5
        if k == len(vs) - 1:
            wv, thv = wv * 0.55, thv * 0.55
        p = 3.2 if inc else 2.6                 # superellipse exponent: rounded rectangle
        ca, sa = np.cos(ang), np.sin(ang)
        a_ = np.sign(ca) * np.abs(ca) ** (2 / p) * 0.5 * wv
        b_ = np.sign(sa) * np.abs(sa) ** (2 / p) * 0.5 * thv
        # labial face convex, lingual flatter (incisors: shovel)
        b_ = np.where(b_ > 0, b_ * (1 + 0.15 * (1 - (2 * a_ / max(wv, 1e-9)) ** 2)), b_)
        h = v * height
        nb = b_ - 0.5 * thv - tilt * h          # labial face at n = 0, lean back toward the root
        P = c[None] + a_[:, None] * t[None] + nb[:, None] * n[None] + (h * up)[None]
        if kind not in ("c", "l") and k == 0:
            P = P + 0.0 * up
        rings.append(P)
    P = np.concatenate(rings)
    tris = []
    for k in range(len(rings) - 1):
        for j in range(NU):
            a, b = k * NU + j, k * NU + (j + 1) % NU
            c2, d = a + NU, b + NU
            tris += [[a, b, d], [a, d, c2]]
    # caps
    n0 = len(P)
    P = np.vstack([P, rings[0].mean(0), rings[-1].mean(0)])
    for j in range(NU):
        tris.append([n0, (j + 1) % NU, j])
        L = (len(rings) - 1) * NU
        tris.append([n0 + 1, L + j, L + (j + 1) % NU])
    return P, np.array(tris)


def _orient(P, tris, up_is_y):
    """flip triangles so normals face away from each tooth's centroid (closed convex-ish meshes)."""
    c = P.mean(0)
    a, b, d = P[tris[:, 0]], P[tris[:, 1]], P[tris[:, 2]]
    nn = np.cross(b - a, d - a)
    out = ((a + b + d) / 3 - c)
    flip = (nn * out).sum(1) < 0
    tris = tris.copy()
    tris[flip] = tris[flip][:, [0, 2, 1]]
    return tris


def _gum(cur, s_list, edge_y, up, th_neck_fn, sc, nprof=8):
    """scalloped bead along the arch at the tooth necks (both sides)."""
    S0, S1 = 0.0, s_list[-1]
    ss = np.linspace(-S1, S1, 72)
    bounds = np.array(s_list)
    rings = []
    for s in ss:
        c, t, n = _at(cur, abs(s))
        if s < 0:
            c = c * np.array([-1, 1, 1]); t = t * np.array([1, 1, -1]); n = n * np.array([-1, 1, 1])
        th = th_neck_fn(abs(s))
        # papilla: the margin dips toward the crown between teeth
        db = np.min(np.abs(bounds - abs(s))) if len(bounds) else 1.0
        pap = np.exp(-(db / (0.0012 * sc)) ** 2)
        neck = edge_y(abs(s)) + up * (0.0 + 0.0)          # neck height (y) passed in edge_y
        yc = neck + up * (0.0016 * sc - 0.0012 * sc * pap)
        ay = 0.0030 * sc                                   # half height of the bead
        an = 0.5 * th + 0.0006 * sc                        # half thickness
        nc = -0.5 * th + 0.0002 * sc
        ph = np.linspace(0, 2 * np.pi, nprof, endpoint=False)
        pr = c[None] + (nc + an * np.cos(ph))[:, None] * n[None] + (np.array([0, 1, 0])[None] * (yc - c[1] + up * ay * np.sin(ph))[:, None])
        rings.append(pr)
    P = np.concatenate(rings)
    tris = []
    for k in range(len(rings) - 1):
        for j in range(nprof):
            a, b = k * nprof + j, k * nprof + (j + 1) % nprof
            tris += [[a, b, b + nprof], [a, b + nprof, a + nprof]]
    tris = np.array(tris)
    # orient: outward from the local ring centre
    rc = np.repeat(np.array([r.mean(0) for r in rings]), nprof, 0)
    a, b, d = P[tris[:, 0]], P[tris[:, 1]], P[tris[:, 2]]
    nn = np.cross(b - a, d - a)
    flip = (nn * ((a + b + d) / 3 - rc[tris[:, 0]])).sum(1) < 0
    tris[flip] = tris[flip][:, [0, 2, 1]]
    return P, tris


def _part(name, mat, P, tris, bone):
    n = len(P)
    UV = np.c_[(P[:, 0] - P[:, 0].min()) / max(np.ptp(P[:, 0]), 1e-9), (P[:, 1] - P[:, 1].min()) / max(np.ptp(P[:, 1]), 1e-9)]
    return Part(name, mat, P, UV, tris, np.tile([bone, 0, 0, 0], (n, 1)), np.tile([1.0, 0, 0, 0], (n, 1)))


def build(base, rig, cav_verts):
    mw = rig.mw
    sc = mw / MW0
    co_u, xr_u, xs, zs = _lining_fit(base, rig, cav_verts, +1)
    co_u = co_u.copy(); co_u[0] -= CLEAR
    sy0 = float(np.interp(0.0, rig.seam_x, rig.seam_y))
    cur_u = _curve(co_u, 2.2 * mw)
    out = {}
    REPORT.clear()
    REPORT["lining_fit_mm"] = [round(float(co_u[0] * 1000), 2), float(co_u[1]), float(co_u[2])]
    for row, spec, up, bone in (("upper", UPPER, 1.0, base.bidx["head"]), ("lower", LOWER, -1.0, base.bidx["jaw"])):
        Ps, Ts, nb = [], [], 0
        s = 0.0
        total = sum(w for _, w, *_ in spec) * sc
        bounds = []
        necks = []
        if row == "upper":
            cur = cur_u
        else:
            # lower arch: inside the upper by the upper incisor's edge thickness + overjet, a bit less at the back
            th_e = UPPER[0][4] * sc
            x, z, ss_ = cur_u
            # offset curve along the inward normal
            off = []
            for si in ss_:
                c, t, n = _at(cur_u, si)
                k = np.clip(si / total, 0, 1)
                d = (th_e + OVERJET * sc + 0.0004 * sc) * (1 - k) + (0.0042 * sc) * k     # posteriors well inside (laugh)
                off.append(c - d * n)
            off = np.array(off)
            keep = off[:, 0] >= 0
            off = off[keep]
            xx, zz = off[:, 0], off[:, 2]
            co_l = _lining_fit(base, rig, cav_verts, -1)[0]
            zz = np.minimum(zz, co_l[0] + co_l[1] * xx ** 2 + co_l[2] * xx ** 4 - CLEAR)   # inside the lower lining
            sss = np.r_[0, np.cumsum(np.hypot(np.diff(xx), np.diff(zz)))]
            cur = (xx, zz, sss)
            # lower teeth are narrower overall: scale widths so the last lower molar ends near the last upper one
        for name, w, hgt, thn, the in spec:
            w, hgt, thn, the = w * sc, hgt * sc, thn * sc, the * sc
            sm = s + 0.5 * w
            c, t, n = _at(cur, sm)
            k = s / total
            spee = SPEE * sc * k ** 1.6
            if row == "upper":
                ey = sy0 - HANG * sc + spee
            else:
                post = np.clip((k - 0.25) / 0.35, 0, 1)
                ey = sy0 - HANG * sc + spee + OVERBITE * sc * (1 - post) + 0.0002 * sc * post
            c = c.copy(); c[1] = ey
            tilt = (0.16 if name in ("c", "l") else 0.06) * (1 if row == "upper" else 0.5)
            ww = w - GAP * sc
            for mir in (1, -1):
                M = np.array([mir, 1, 1])
                cc, tt, nn = c * M, t * np.array([1, 1, mir]), n * M
                P, T = _tooth(cc, tt, nn, np.array([0, up, 0]), ww, hgt, thn, the, name, tilt)
                T = _orient(P, T, True)
                Ps.append(P); Ts.append(T + nb); nb += len(P)
            bounds.append(s)
            necks.append((sm, ey + up * hgt * 0.92, thn))
            s += w
        bounds.append(s)
        P, T = np.concatenate(Ps), np.concatenate(Ts)
        out[f"teeth_{row}"] = _part(f"teeth_{row}", "teeth", P, T, bone)
        ns = np.array([q[0] for q in necks]); ny = np.array([q[1] for q in necks]); nt = np.array([q[2] for q in necks])
        GP, GT = _gum(cur, bounds, lambda si: float(np.interp(si, ns, ny)), up, lambda si: float(np.interp(si, ns, nt)), sc)
        out[f"gums_{row}"] = _part(f"gums_{row}", "gums", GP, GT, bone)
        REPORT[row] = dict(teeth=len(spec) * 2, tris=int(len(T)), gum_tris=int(len(GT)), arc_mm=round(s * 1000, 1),
                           last_x_mm=round(float(_at(cur, s)[0][0] * 1000), 1), last_z_mm=round(float(_at(cur, s)[0][2] * 1000), 1),
                           edge_y_mm=round(float((sy0 - HANG * sc) * 1000), 2))
    return [out["teeth_upper"], out["teeth_lower"], out["gums_upper"], out["gums_lower"]], REPORT
