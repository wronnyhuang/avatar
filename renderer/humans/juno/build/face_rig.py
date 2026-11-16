"""asset — face_shapes helper: landmark/topology analysis of the face on ANY Juno base (procedural, no coordinates).

FaceRig(base) measures, from the current base mesh + MH expression targets:
  jaw      hinge, world opening axis (= jaw bone local +x), per-vertex jaw weight w_jaw (1 = rides the jaw: chin, lower
           lip; 0 = upper lip / skull) derived from MH's own `mouth-open` unit (its artist falloff), smoothed.
  lips     seam pairs (upper inner-lip vertex, lower inner-lip vertex that touches it at rest), mouth corners,
           half-width mw, the seam line y(x)/z(x); distance-to-seam field.
  eyes     per side: centre c, radius R, upper-lid weight a_up (from MH eye-closure unit), lower-lid weight a_lo
           (procedural), the aperture elevations phi_up(az)/phi_lo(az) (lid margins, radians, about the eye centre in
           the y-z plane), canthi.
  misc     interocular distance iod, cheek centres, alar (nose wing) points, skin graph smoothing.
All lengths scale with iod / mw / R so the lead's proportion changes carry through.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from scipy.spatial import cKDTree

from juno_base import smoothstep

# Lid clearance: closed / moving lids keep at least the eyes_geo cornea radius (measured per eye from the fitted ball,
# eyes_geo.EyeFit.cornea_radius) + LID_CLEAR, and never less than eyes_geo.conform_lids' sphere (CONFORM_K * R), so the
# build's conform pass has nothing to change on face_shapes' lid shapes.
LID_CLEAR = 0.00035
CONFORM_K = 1.036


def lid_floor(nz, R, fit=None):
    """minimum lid radius at direction cos(angle from the gaze axis) = nz. fit = eyes_geo.EyeFit of that eye."""
    if fit is None:
        cr = R * 1.004
    else:
        cr = fit.cornea_radius(np.arccos(np.clip(nz, -1, 1)))
    return np.maximum(cr + LID_CLEAR, CONFORM_K * R)


def rotate(P, center, axis, ang):
    """Rodrigues: rotate points P (n,3) about axis (unit, 3) through center by per-point angle ang (n,) radians."""
    q = P - center
    k = axis / np.linalg.norm(axis)
    c, s = np.cos(ang)[:, None], np.sin(ang)[:, None]
    return center + q * c + np.cross(k, q) * s + k * (q @ k)[:, None] * (1 - c)


def gauss(P, c, sig):
    return np.exp(-((P - c) ** 2).sum(1) / (2 * sig * sig))


class FaceRig:
    def __init__(self, base):
        self.base = base
        VA, skin = base.VA, base.skin_mask
        self.VA, self.skin, self.nV = VA, skin, base.nV
        self.skinf = skin.astype(float)
        # skin graph (row-normalised adjacency, skin-only)
        t = base.tris
        e = np.concatenate([t[:, [0, 1]], t[:, [1, 2]], t[:, [2, 0]]])
        e = e[skin[e[:, 0]] & skin[e[:, 1]]]
        A = sp.coo_matrix((np.ones(len(e) * 2), (np.r_[e[:, 0], e[:, 1]], np.r_[e[:, 1], e[:, 0]])),
                          shape=(self.nV, self.nV)).tocsr()
        A.data[:] = 1.0
        deg = np.asarray(A.sum(1)).ravel()
        self.Adj = sp.diags(1 / np.maximum(deg, 1)) @ A
        self.has_nb = deg > 0
        lm = base.lm
        self.iod = float(np.linalg.norm(lm["eye_c_l"] - lm["eye_c_r"]))
        self._jaw()
        self._lips()
        self._eyes()
        self._misc()

    # ---------------------------------------------------------------- utils
    def smooth(self, f, n=2, keep=None):
        f = np.asarray(f, float).copy()
        for _ in range(n):
            g = self.Adj @ f
            g = np.where(self.has_nb[:, None] if f.ndim == 2 else self.has_nb, g, f)
            f = 0.5 * f + 0.5 * g
            if keep is not None:
                f[keep[0]] = keep[1]
        return f

    def ut(self, name):
        return self.base.target(f"expression/units/caucasian/{name}.target.gz")

    # ---------------------------------------------------------------- jaw
    def _jaw(self):
        b, VA = self.base, self.VA
        self.hinge = b.lm["jaw_hinge"].copy()
        ax = b.rest[b.bidx["jaw"]][:3, 0].astype(float)
        self.jaw_axis = ax / np.linalg.norm(ax)
        ay = b.rest[b.bidx["jaw"]][:3, 2].astype(float)
        self.jaw_side_axis = ay / np.linalg.norm(ay)       # jaw local +z (mostly down/back): side swing axis
        d = self.ut("mouth-open")
        q = VA - self.hinge
        tng = np.cross(self.jaw_axis, q)
        a = (d * tng).sum(1) / np.maximum((tng * tng).sum(1), 1e-12)   # radians of opening that best explains d
        a[~self.skin] = 0
        mag = np.linalg.norm(d, axis=1)
        big = self.skin & (mag > 0.6 * mag[self.skin].max())
        self.a_ref = float(np.median(a[big]))
        w = np.clip(a / self.a_ref, 0, 1)
        w[~self.skin] = 0
        self.w_jaw = np.clip(self.smooth(w, 2), 0, 1)

    # ---------------------------------------------------------------- lips
    def _lips(self):
        VA, skin, w = self.VA, self.skin, self.w_jaw
        mo = self.base.lm["mouth"]
        near = skin & (np.linalg.norm(VA - mo, axis=1) < 0.7 * self.iod)
        U = np.where(near & (w < 0.42))[0]
        L = np.where(near & (w > 0.58))[0]
        dd, j = cKDTree(VA[L]).query(VA[U])
        ok = (dd < 0.0016) & (w[L[j]] - w[U] > 0.2)
        su, sl = U[ok], L[j[ok]]
        sP = 0.5 * (VA[su] + VA[sl])
        # keep the front-most contact pair per 1.5 mm x-bin (the mouth bag behind can touch too)
        keep = []
        xb = np.round(sP[:, 0] / 0.0015).astype(int)
        for b_ in np.unique(xb):
            k = np.where(xb == b_)[0]
            keep.append(k[np.argmax(sP[k, 2])])
        keep = np.array(keep)
        su, sl, sP = su[keep], sl[keep], sP[keep]
        o = np.argsort(sP[:, 0])
        self.seam_u, self.seam_l, sP = su[o], sl[o], sP[o]
        self.seamP = sP
        self.corner = {"l": sP[np.argmax(sP[:, 0])].copy(), "r": sP[np.argmin(sP[:, 0])].copy()}
        self.mw = float(0.5 * (self.corner["l"][0] - self.corner["r"][0]))
        # seam line as function of x (front-most per bin)
        xs = np.linspace(-self.mw, self.mw, 25)
        ys, zs = [], []
        for x in xs:
            k = np.abs(sP[:, 0] - x) < 0.12 * self.mw
            if not k.any():
                k = np.argsort(np.abs(sP[:, 0] - x))[:2]
            pz = sP[k]
            f = pz[np.argmax(pz[:, 2])]
            ys.append(f[1]); zs.append(f[2])
        self.seam_x, self.seam_y, self.seam_z = xs, np.array(ys), np.array(zs)
        allseam = np.r_[self.seam_u, self.seam_l]
        self.d_seam, _ = cKDTree(VA[allseam]).query(VA)
        x = VA[:, 0]
        self.u = x / self.mw                                    # horizontal, 0 centre, +-1 at corners
        sy = np.interp(np.clip(x, -self.mw, self.mw), xs, self.seam_y)
        sz = np.interp(np.clip(x, -self.mw, self.mw), xs, self.seam_z)
        self.h = (VA[:, 1] - sy) / self.mw                      # height above the lip line (mw units)
        self.dz = (VA[:, 2] - sz) / self.mw
        self.upper = (w < 0.5) & skin & (self.h > -0.8)     # r3: not the neck/body below the mouth (w_jaw = 0 there)
        self.lower = (w >= 0.5) & skin
        mc = self.seamP.mean(0)
        self.mouth_c = np.array([0.0, mc[1], self.seam_z.max()])
        # lip (vermilion + a bit) mask: near the seam, in front of the mouth bag
        self.lipmask = (np.exp(-(self.d_seam / (0.42 * self.mw)) ** 2) * skin
                        * smoothstep(-0.55, -0.15, self.dz))

    # ---------------------------------------------------------------- eyes
    def _eyes(self):
        """Lid analysis on the lead's lid-margin vertices (juno_sculpt.lid_margins, from the MH lash cards)."""
        from juno_sculpt import lid_margins
        import eyes_geo
        VA, skin = self.VA, self.skin
        self.eyes = {}
        for side in ("l", "r"):
            c, R = self.base.eye[side]
            c = np.asarray(c, float)
            fit = eyes_geo.EyeFit(self.base, {}, side)
            d = VA - c
            r = np.linalg.norm(d, axis=1)
            el = np.arctan2(d[:, 1], d[:, 2])
            az = np.arctan2(d[:, 0] * (1 if side == "l" else -1), d[:, 2])   # + = outward (lateral)
            up, lo = lid_margins(self.base, side)
            # aperture profiles phi(az) along the margins
            ou, ol = np.argsort(az[up]), np.argsort(az[lo])
            azu, elu = az[up][ou], el[up][ou]
            azl, ell = az[lo][ol], el[lo][ol]
            a0, a1 = min(azu[0], azl[0]), max(azu[-1], azl[-1])
            azb = np.linspace(a0, a1, 41)
            pu = np.interp(azb, azu, elu)
            pl = np.interp(azb, azl, ell)
            mid = 0.5 * (pu + pl)
            # canthi: the margins meet
            pu[0] = pl[0] = mid[0]; pu[-1] = pl[-1] = mid[-1]
            azc = np.clip(az, a0, a1)
            phu, phl = np.interp(azc, azb, pu), np.interp(azc, azb, pl)
            phm = 0.5 * (phu + phl)
            region = skin & (r < 2.3 * R) & (d[:, 2] > -0.5 * R) & (np.abs(az) < np.radians(100))
            # canthus taper in azimuth (soft beyond the corners)
            span = a1 - a0
            taper = (smoothstep(a0 - 0.18 * span, a0 + 0.04 * span, az)
                     * (1 - smoothstep(a1 - 0.04 * span, a1 + 0.18 * span, az)))
            above = el >= phm
            # upper lid shell: margin .. crease rides fully, fades to the brow
            t_up = el - phu
            a_up = (1 - smoothstep(np.radians(22), np.radians(48), t_up)) * (1 - smoothstep(1.45 * R, 1.9 * R, r))
            a_up = a_up * region * above
            # lower lid: margin .. lid-cheek junction
            t_lo = phl - el
            a_lo = (1 - smoothstep(np.radians(8), np.radians(36), t_lo)) * (1 - smoothstep(1.3 * R, 1.8 * R, r))
            a_lo = a_lo * region * ~above
            a_up = np.clip(self.smooth(a_up, 2), 0, 1) * region * above
            a_lo = np.clip(self.smooth(a_lo, 2), 0, 1) * region * ~above
            a_up[up] = 1.0
            a_lo[lo] = 1.0
            self.eyes[side] = dict(c=c, R=R, d=d, r=r, el=el, az=az, a_up=a_up, a_lo=a_lo, phu=phu, phl=phl,
                                   azb=azb, pu=pu, pl=pl, taper=taper, region=region, up=up, lo=lo,
                                   floor=(lambda nz, R=R, fit=fit: lid_floor(nz, R, fit)))

    def lid_delta(self, side, which, dphi, P0=None, path=False):
        """Rotate a lid about the eye centre (eye-space x axis). `which` in up/lo; dphi = callable(az) or (nV,) array of
        margin angles (radians, + = DOWN). Returns the delta at weight 1, or (A, B) if path=True with the linear+square
        path w*A + w^2*B hitting the true arc at w = 0, .5, 1. The radius rides the cornea profile: skin near the ball
        gains the floor increase it sweeps over (keeps lid thickness), and never ends inside lid_floor."""
        E = self.eyes[side]
        c, R = E["c"], E["R"]
        P0 = self.VA if P0 is None else P0
        a = E["a_up"] if which == "up" else E["a_lo"]
        ang = a * (dphi(E["az"]) if callable(dphi) else dphi) * E["taper"]
        mg = E["up"] if which == "up" else E["lo"]
        ang = self.smooth(ang, 14, keep=(mg, ang[mg].copy())) * E["region"] * smoothstep(0.0, 0.08, a)   # no shear crease at the sharp canthi
        sel = np.where(np.abs(ang) > 1e-5)[0]
        if not len(sel):
            Z = np.zeros((self.nV, 3))
            return (Z, Z.copy()) if path else Z
        q0 = P0[sel] - c
        r0 = np.linalg.norm(q0, axis=1)
        f0 = E["floor"](q0[:, 2] / np.maximum(r0, 1e-9))
        near = 1 - smoothstep(1.15 * R, 1.45 * R, r0)       # skin on the ball: carries the bulge increase
        outside = r0 >= f0 - LID_CLEAR                       # front skin (not the conjunctival tuck)

        def at(t):
            p = rotate(P0[sel], c, np.array([1.0, 0, 0]), ang[sel] * t)
            q = p - c
            nz = q[:, 2] / np.maximum(r0, 1e-9)
            f1 = E["floor"](nz)
            need = r0 + near * np.maximum(0, f1 - f0)
            need = np.where(outside & (r0 < 1.6 * R), np.maximum(need, f1), need)
            return c + q / np.maximum(r0, 1e-9)[:, None] * need[:, None]

        D1 = at(1.0) - P0[sel]
        if not path:
            return _full(self.nV, sel, D1)
        Dm = at(0.5) - P0[sel]
        Bq = 2 * D1 - 4 * Dm
        return _full(self.nV, sel, D1 - Bq), _full(self.nV, sel, Bq)

    def seal(self, D, P0=None):
        """Push any front skin that a (weight-1) delta D moves inside the cornea profile back out radially."""
        P0 = self.VA if P0 is None else P0
        D = D.copy()
        for side in ("l", "r"):
            E = self.eyes[side]
            c, R = E["c"], E["R"]
            # lid zone only (r3: the region reaches the mouth corners on the fitted head, where the cornea profile's
            # extrapolation threw cavity-lining vertices ~80 mm forward)
            k = np.where(E["region"] & (np.abs(D).sum(1) > 0) & (E["r"] < 1.3 * R) & (E["d"][:, 2] > 0))[0]
            if not len(k):
                continue
            q0 = P0[k] - c
            r0 = np.linalg.norm(q0, axis=1)
            out0 = r0 >= E["floor"](q0[:, 2] / np.maximum(r0, 1e-9)) - LID_CLEAR
            q1 = q0 + D[k]
            r1 = np.linalg.norm(q1, axis=1)
            f1 = E["floor"](q1[:, 2] / np.maximum(r1, 1e-9))
            bad = out0 & (r1 < f1)
            kk = k[bad]
            D[kk] = c + q1[bad] / r1[bad, None] * f1[bad, None] - P0[kk]
        return D

    def phi(self, side, which):
        """callable az -> margin elevation (radians) of the upper / lower lid."""
        E = self.eyes[side]
        prof = E["pu"] if which == "up" else E["pl"]
        return lambda az: np.interp(az, E["azb"], prof)

    def aperture(self, side):
        E = self.eyes[side]
        return lambda az: np.interp(az, E["azb"], E["pu"] - E["pl"])

    # ---------------------------------------------------------------- misc
    def _misc(self):
        VA, skin = self.VA, self.skin
        lm = self.base.lm
        nt = lm["nose_tip"]
        # alar (nose wing) points: widest skin points in the lower nose band
        band = skin & (VA[:, 1] < nt[1] + 0.002) & (VA[:, 1] > nt[1] - 0.25 * self.iod) & \
            (VA[:, 2] > nt[2] - 0.32 * self.iod) & (np.abs(VA[:, 0]) < 0.45 * self.iod)
        self.alar = {}
        for side, sg in (("l", 1), ("r", -1)):
            k = np.where(band & (sg * VA[:, 0] > 0))[0]
            v = k[np.argmax(sg * VA[k, 0] + 0.3 * VA[k, 2])]
            self.alar[side] = VA[v].copy()
        self.cheek = {}
        for side in ("l", "r"):
            e = lm[f"eye_c_{side}"]
            tgt = np.array([e[0] * 1.08, e[1] - 0.48 * (e[1] - self.mouth_c[1]), e[2] + 0.5 * self.iod])
            k = np.where(skin & (np.abs(VA[:, 1] - tgt[1]) < 0.06 * self.iod) & (np.abs(VA[:, 0] - tgt[0]) < 0.06 * self.iod))[0]
            v = k[np.argmax(VA[k, 2])]
            self.cheek[side] = VA[v].copy()

    def nasolabial(self, side):
        """0 medial (lip side) -> 1 lateral (cheek side) of the alar->corner(+out) fold line, soft over ~2.5 mm."""
        a = self.alar[side]
        cc = self.corner[side] + np.array([(1 if side == "l" else -1) * 0.2 * self.mw, -0.25 * self.mw, 0])
        dvec = (cc - a)[:2]
        nrm = np.array([dvec[1], -dvec[0]])
        nrm /= np.linalg.norm(nrm)
        if (nrm[0] > 0) != (side == "l"):
            nrm = -nrm
        s = (self.VA[:, :2] - a[:2]) @ nrm
        return smoothstep(-0.1 * self.mw, 0.12 * self.mw, s)


def _full(nV, sel, d):
    D = np.zeros((nV, 3))
    D[sel] = d
    return D
