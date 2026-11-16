"""HAIR — junor scalp groom: a side-parted, soft-waved lob (just below the jaw), left side tucked behind the ear.

    raw = build(H, rng, n=120000)   # H = head.Head(asset) -> {"scalp": {...}, "fuzz": {...}} (lod0 + lod1 arrays)

Pipeline (all vectorised numpy, every strand marched — no guide interpolation, so nothing is ever blended across the
part):
  1 scalp region    an analytic hairline in the head frame (azimuth/height table from the eye line, ears carved out,
                    a few mm of positional noise + a density ramp: no hard helmet edge), area-uniform roots.
  2 comb field      side part at PART_AZ (her left): the big side sweeps across the crown to her right, the small side
                    goes down-back and TUCKS behind the left ear, the back fans out from a crown whorl.
  3 march           every strand: root lift off the scalp (volume, no helmet), gravity bending against stiffness,
                    collision with a growing shell over the head (layers ride on layers) + neck/clothes, stopped at
                    a per-strand tip line (lob length, softly layered: crown strands shorter).
  4 shape           coherent S-waves per lock below the ear line, 3 clump scales (locks / sub-locks / strand groups)
                    pulling toward their centres more toward the tips, per-strand frizz, a few strays, collision again.
  5 fuzz            baby hairs along the hairline band + a few tapered flyaways off the outer layer.
Everything scales with the head (H.u = eye line -> crown, H.d = eye line -> chin), so it re-runs on any head build.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from head import nrm, smoothstep, resample

P_SCALP, P_FUZZ = 32, 12
# style presets (regroom.py --style): "lob" = chin-length A-line lob (r1-r6); "waves" = loose shoulder-length waves
STYLES = {
    "lob": dict(len_front=0.12, len_back=0.12, lam=(0.085, 0.03), amp=(0.0055, 0.0045), h_cap=0.033, curl=0.007,
                bulge=0.004, wave_top=0.45, tuck=0.0, fall=1.0, lift=(0.5, 0.7), coherent=True),
    "waves": dict(len_front=0.95, len_back=0.30, lam=(0.080, 0.030), amp=(0.009, 0.006), h_cap=0.040, curl=0.004,
                  bulge=0.006, wave_top=0.20, tuck=1.0, fall=0.0, lift=(0.45, 0.6)),
}
STYLE = dict(STYLES["lob"])
PART_AZ = 20.0            # deg, + = her left: the part sits over her left eye's outer half
HAIR_ROOT = [0.034, 0.020, 0.013]   # linear RGB: rich dark brown (not black)
HAIR_TIP = [0.072, 0.044, 0.028]    # sun-lightened warm ends

# hairline: |azimuth| deg -> height above the eye line (in H.u when >= 0, in H.d below the eye line)
HL_AZ = np.array([0, 8, 18, 28, 36, 44, 52, 60, 68, 76, 82, 95, 115, 140, 165, 180], float)
HL_T = np.array([.585, .595, .605, .62, .635, .615, .50, .33, .14, -.08, -.26, -.42, -.58, -.86, -1.0, -1.0])   # soft widow's peak + M temple recession


class Groom:
    def __init__(self, H, rng):
        self.H, self.rng = H, rng
        self.d = float(H.eye_y - H.chin[1])            # eye line -> chin
        self.k = H.u / 0.125                           # scale vs a realistic adult head (eye->crown 12.5 cm)

    # ------------------------------------------------------------------ 1 scalp region
    def hairline_y(self, az):
        t = np.interp(np.abs(az), HL_AZ, HL_T)
        return self.H.eye_y + np.where(t >= 0, t * self.H.u, t * self.d)

    def ear_excl(self, X, side, margin):
        e = self.H.ear[side]
        cy = 0.5 * (e["top"] + e["bot"]); hy = 0.5 * (e["top"] - e["bot"]) + margin
        cz = 0.5 * (e["front"] + e["back"]); hz = 0.5 * (e["front"] - e["back"]) + margin
        sgn = 1.0 if side == "l" else -1.0
        q = ((X[..., 1] - cy) / hy) ** 2 + ((X[..., 2] - cz) / hz) ** 2
        return (q < 1.0) & (sgn * X[..., 0] > 0.0), q

    def density(self, X, noise=True):
        H = self.H
        az, el, r = H.az_el(X)
        y_hl = self.hairline_y(az)
        if noise:
            a = np.radians(az)
            y_hl = y_hl + 0.0045 * self.k * (np.sin(a * 17.0 + 0.7) * 0.6 + np.sin(a * 41.0 + 2.1) * 0.45
                                             + np.sin(a * 97.0 + 4.0) * 0.3 + np.sin(a * 211.0 + 1.3) * 0.2)
        dy = X[..., 1] - y_hl
        if STYLE.get('fall', 0.0) > 0:      # hair covers the hairline now: a tighter ramp (no grey see-through band)
            dens = smoothstep(-0.003 * self.k, 0.009 * self.k, dy) ** 1.3
        else:
            dens = smoothstep(-0.004 * self.k, 0.015 * self.k, dy) ** 1.6
        for s in ("l", "r"):
            inside, q = self.ear_excl(X, s, 0.007 * self.k)
            dens = dens * smoothstep(1.0, 1.6, q)            # nothing on the ear, a soft ramp around it
        dens = dens * (r < H.R * 1.6)
        return dens, dy

    def roots(self, n):
        H, rng = self.H, self.rng
        cand, cn = H.sample(int(n * 3.2), rng)
        dens, dy = self.density(cand)
        keep = rng.random(len(cand)) < dens
        X, N, dy, dens = cand[keep], cn[keep], dy[keep], dens[keep]
        if len(X) > n:
            i = rng.choice(len(X), n, replace=False)
            X, N, dy, dens = X[i], N[i], dy[i], dens[i]
        return X, N, dy, dens

    # ------------------------------------------------------------------ 2 comb field
    def part_x(self, z):
        H, k = self.H, self.k
        z = np.asarray(z, float)
        # irregular part: a slow wander (+-1.5 mm) + a zig-zag (+-1.4 mm, ~7 mm period) where locks interleave
        w = (z - H.O[2]) / (0.007 * k)
        tri = 2.0 * np.abs((w + 0.37 * np.sin(1.7 * w)) % 2.0 - 1.0) - 1.0
        wander = 0.0015 * k * (np.sin(z / (0.021 * k) + 0.6) * 0.7 + np.sin(z / (0.009 * k) + 2.2) * 0.3)
        return H.O[0] + H.R * np.sin(np.radians(PART_AZ)) + 0.0014 * k * tri + wander

    def labels(self, X):
        """0 = big side (her right of the part), 1 = small side (her left), 2 = back (behind the crown whorl)"""
        H = self.H
        zw = H.O[2] - 0.30 * H.R
        lab = np.where(X[:, 0] > self.part_x(X[:, 2]), 1, 0)
        back = X[:, 2] < zw
        lab = np.where(back, 2, lab)
        return lab

    def comb(self, X, N, lab):
        H = self.H
        W = H.O + H.R * nrm(np.array([0.10, 0.80, -0.45]))         # crown whorl
        zw = H.O[2] - 0.30 * H.R
        fwd = smoothstep(zw - 0.10 * H.R, zw + 0.25 * H.R, X[:, 2])     # 0 behind the whorl .. 1 in front
        fall = STYLE.get('fall', 0.0)
        big = np.array([-1.0, -0.10 - 0.22 * fall, 0.12])      # across the crown, landing on her right temple (face-framing)
        small = nrm(np.array([1.0, -0.35, -0.65])) * (1 - fall) + fall * nrm(np.array([1.0, -0.6, 0.15]))  # left: tuck / fall over the temple
        radial = nrm(X - W)
        # no hole at the whorl: close to it the strands lean the same way (down-back) instead of fanning out
        dw = np.linalg.norm(X - W, axis=1)
        radial = nrm(radial + (1.5 * smoothstep(0.03 * self.k, 0.0, dw))[:, None] * np.array([-0.2, -0.3, -1.0]))
        d = np.where(lab[:, None] == 0, big, small)
        d = nrm(d) * fwd[:, None] + radial * (1.0 - fwd[:, None])
        # on the forward-facing forehead the strands first rise up and back over the hairline (the swoop), then
        # the field above takes them sideways
        front = smoothstep(0.30, 0.70, N[:, 2]) * smoothstep(H.eye_y, H.eye_y + 0.3 * H.u, X[:, 1])
        fall = STYLE.get('fall', 0.0)
        front = front * (1.0 - 0.85 * fall)          # fall: no slicked-back swoop (hair drapes from the part)
        d = d * (1.0 - 0.7 * front[:, None]) + front[:, None] * np.array([0.0, 0.8, -0.35])
        # where the surface turns to the side the sideways comb vector would project to ~0 (a pinch): blend toward
        # gravity there so the field stays continuous (over the top -> down the side)
        up = smoothstep(0.05, 0.55, N[:, 1])
        # fall: on the forehead the sweep stays lateral (a side sweep over the hairline, not bangs down the face)
        fr0 = smoothstep(0.30, 0.70, N[:, 2]) * smoothstep(H.eye_y + 0.25 * H.u, H.eye_y + 0.5 * H.u, X[:, 1])
        up = np.maximum(up, fall * fr0)[:, None]
        d = d * up + np.array([0.0, -1.0, 0.0]) * (1.0 - up) + 0.25 * d * (1.0 - up)
        t = d - (d * N).sum(1, keepdims=True) * N
        bad = (t * t).sum(1) < 1e-4
        t[bad] = np.array([0.0, -1.0, 0.0]) - N[bad, 1:2] * N[bad]
        return nrm(t)

    # ------------------------------------------------------------------ 3 march
    def march(self, X0, N0, lab, dy, step=0.005):
        H, rng, k = self.H, self.rng, self.k
        S = len(X0)
        az, el, _ = H.az_el(X0)
        t0 = self.comb(X0, N0, lab)
        # flow variation: every strand combs a few degrees off the field, ~2 % cross over at a steep angle
        # (strand-level breakup of the sheet, crossing strands on top of the mass)
        ang = np.radians(rng.normal(0.0, 5.0, S) + (rng.random(S) < 0.02) * rng.normal(0.0, 25.0, S))
        tan_a = np.tan(np.clip(ang, -1.0, 1.0))
        t0 = nrm(t0 + tan_a[:, None] * nrm(np.cross(N0, t0)))
        # root lift: more at the part and the crown (volume), less at the nape / hairline
        near_part = np.exp(-((X0[:, 0] - self.part_x(X0[:, 2])) / (0.025 * k)) ** 2) * (lab < 2)
        crown = smoothstep(20, 70, el)
        l0, l1 = STYLE.get('lift', (0.45, 0.6))
        lift = (l0 + l1 * crown) * np.where(lab == 0, 1.0 + 0.1 * near_part * STYLE.get('fall', 0.0), 1.0 + 0.2 * near_part * STYLE.get('fall', 0.0))                      # 0..~1.4: shell height scale (volume)
        lift *= smoothstep(-0.002 * k, 0.02 * k, dy) * 0.8 + 0.2       # hairline strands lie flatter
        lift *= 0.8 + 0.4 * rng.random(S)
        Wh = H.O + H.R * nrm(np.array([0.10, 0.80, -0.45]))
        lift *= 0.35 + 0.65 * smoothstep(0.0, 0.035 * k, np.linalg.norm(X0 - Wh, axis=1))   # flat at the whorl (no hole)
        D = nrm(t0 + N0 * 0.25)
        # tip line: lob just below the jaw, longer at the front (A-line), crown layers shorter
        side = np.abs(np.sin(np.radians(az)))
        backness = smoothstep(90, 180, np.abs(az))
        y_end = H.chin[1] - self.d * (STYLE['len_front'] - STYLE['len_back'] * backness + 0.04 * side)     # A-line: shorter at the nape (collar clearance)
        y_end = y_end + self.d * 0.35 * smoothstep(35, 80, el) * rng.random(S) ** 2     # soft layers
        # ragged, layered ends: each lock ends at its own length (+-3 cm), strands +-0.7 cm around it
        if getattr(self, "mid_pre", None) is not None:
            nl = int(self.mid_pre.max()) + 1
            y_end = y_end + self.d * 0.12 * rng.standard_normal(nl)[self.mid_pre]
        y_end = y_end + self.d * 0.035 * rng.standard_normal(S)
        # tuck: small-side strands in front of / above the left ear go back over it
        e = H.ear["l"]
        tuck = (STYLE.get('tuck', 1.0) * (lab == 1) * smoothstep(e["back"] - 0.012 * k, e["back"] + 0.006 * k, X0[:, 2])
                * smoothstep(1.0, 0.75, (X0[:, 1] - e["top"]) / (0.7 * H.u)) * (X0[:, 1] > e["bot"]))
        # a few face-framing strands at her left temple stay OUT of the tuck (fall in front of the ear)
        frame = (lab == 1) & (az > 50) & (az < 64) & (dy < 0.02 * k) & (rng.random(S) < 0.35)
        tuck = tuck * ~frame
        Btuck = np.array([e["x"] - 0.006 * k, e["top"] + 0.006 * k, e["back"] - 0.012 * k])
        # stiffness: gravity bend per metre; long free lengths sag, short roots stand
        kg = (14.0 + 6.0 * rng.random(S)) / k
        h_root = (0.004 + 0.015 * lift) * k                             # root lift: the shell height (volume)
        # the volume builds up away from the hairline (no hair "brim" a few mm behind the forehead edge)
        h_root = h_root * (0.35 + 0.65 * smoothstep(0.0, 0.025 * k, dy))
        layer = (0.03 + 0.04 * rng.random(S))                           # shell growth per metre of strand
        h_cap = STYLE['h_cap'] * k
        nmax = int(np.ceil((H.u * 2.2 + self.d * 1.8) / step))
        Xs = np.zeros((S, nmax + 1, 3)); Xs[:, 0] = X0
        p = X0 + N0 * 0.0003
        alive = np.ones(S, bool)
        s_arc = np.zeros(S)
        nstep = np.zeros(S, int)
        down = np.array([0.0, -1.0, 0.0])
        passed = np.zeros(S, bool)
        L_rest = (H.u * 1.2 + (H.chin[1] - y_end).clip(0) + 0.6 * self.d)        # planned length (m), rough
        y_shell = H.O[1] - 0.15 * H.R          # above this the comb holds hair on its shell; below it hangs free
        sdp, nrp = H.sd(p)
        for i in range(nmax):
            a = alive
            pa = p[a]
            sd0, n0 = sdp[a], nrp[a]
            on = (pa[:, 1] > y_shell) & (sd0 < h_cap + 0.012 * k)
            ta = tuck[a] * (~passed[a])
            # gravity: weak while the comb holds the strand on the head, full once it hangs free
            g = (kg[a] * step * smoothstep(0.0, 0.012 * k, s_arc[a]) * np.where(on, 0.35, 3.0) * (1.0 - 0.9 * ta))[:, None]
            Dn = D[a] + g * down
            # comb steering: on the head the strand keeps following the comb field (over the top, then down the side)
            cf = self.comb(pa, n0, lab[a])
            cf = nrm(cf + tan_a[a][:, None] * nrm(np.cross(n0, cf)))
            Dn = Dn + (np.where(on, 0.30, 0.0) * (1.0 - ta))[:, None] * cf
            if ta.any():
                Dn = Dn + (ta * 1.2)[:, None] * nrm(Btuck - pa)
            # hanging part: a soft inward curve toward the neck (rounded lob, not a skirt)
            hz = pa - H.O; hz[:, 1] = 0.0
            Dn = Dn - ((~on) * 0.03 * smoothstep(H.chin[1] + 0.3 * self.d, H.chin[1] - 0.3 * self.d, pa[:, 1]))[:, None] * nrm(hz)
            # hanging hair behind the ears drifts a little back (soft profile, no vertical curtain edge)
            bk = (~on) * smoothstep(H.ear["l"]["front"], H.ear["l"]["back"] - 0.02 * k, pa[:, 2])
            Dn = Dn + (0.05 * bk)[:, None] * np.array([0.0, 0.0, -1.0])
            Dn = nrm(Dn)
            q = p[a] + Dn * step
            # shell height: ramps up over the first ~1.5 cm (root lift), then grows slowly (layers over layers)
            h = np.minimum(h_root[a] * smoothstep(0.0, 0.015 * k, s_arc[a] + step) + layer[a] * s_arc[a], h_cap)
            sd, n = H.sd(q)
            hold = (q[:, 1] > y_shell) & (sd < h + 0.012 * k) & (sd > h)
            corr = np.where(hold, 0.7 * (h - sd), np.maximum(h - sd, 0.0))
            q = q + corr[:, None] * n
            if (q[:, 1] < H.chin[1] + 0.5 * self.d).any():
                q = H.push(q, h, iters=1, body=0.016 * k)
            sdp[a] = sd + corr; nrp[a] = n
            D[a] = nrm(q - p[a])
            p[a] = q
            s_arc[a] += step
            Xs[a, i + 1] = q
            nstep[a] = i + 1
            passed[a] |= q[:, 2] < Btuck[2]
            # clean tuck: a tucked strand never re-enters the box in front of the ear's back edge (no veil on the ear)
            tk = (tuck[a] > 0.2) & passed[a] & (q[:, 1] < e["top"] + 0.012 * k) & (q[:, 1] > e["bot"] - 0.02 * k) & (q[:, 0] > 0)
            if tk.any():
                q[tk, 2] = np.minimum(q[tk, 2], e["back"] - 0.004 * k)
                Xs[np.where(a)[0][tk], i + 1] = q[tk]; p[np.where(a)[0][tk]] = q[tk]
            done = a.copy()
            dn = q[:, 1] < y_end[a]
            low = q[:, 1] < H.chin[1]
            if low.any():                       # resting on the shoulder / collar: the strand ends there (no sliding)
                sb, _ = H.sd_body(q[low])
                dl = dn[low]; dl |= (sb < 0.004 * k) & (s_arc[a][low] > L_rest[a][low]); dn[low] = dl   # only a strand pressed onto the shoulder past its length
            done[a] = dn
            alive &= ~done
            if not alive.any():
                break
        self.stuck = alive.copy()
        last = nstep.copy()
        idx = np.minimum(np.arange(Xs.shape[1])[None, :], last[:, None])
        Xs = np.take_along_axis(Xs, idx[:, :, None], axis=1)
        self.length = s_arc
        self.near_part = near_part
        return Xs[:, :max(last.max(), 2) + 1], y_end

    # ------------------------------------------------------------------ 4 shape
    def frame_along(self, X):
        T = np.gradient(X, axis=1)
        T = nrm(T)
        out = X - self.H.O
        out[..., 1] = 0.0
        out = nrm(out)
        B = nrm(np.cross(T, out))
        Nn = nrm(np.cross(B, T))
        return T, B, Nn

    def clump_ids(self, X0, lab, n):
        rng = self.rng
        ids = np.zeros(len(X0), int)
        base = 0
        for l in range(3):
            m = np.where(lab == l)[0]
            if len(m) == 0:
                continue
            nl = max(1, int(round(n * len(m) / len(X0))))
            seeds = m[rng.choice(len(m), min(nl, len(m)), replace=False)]
            _, j = cKDTree(X0[seeds]).query(X0[m])
            ids[m] = base + j
            base += len(seeds)
        return ids, base

    def pull(self, X, ids, nc, k_tip, start=0.15, power=1.0):
        S, P, _ = X.shape
        cnt = np.bincount(ids, minlength=nc).astype(float)
        C = np.zeros((nc, P, 3))
        np.add.at(C, ids, X)
        C /= np.maximum(cnt, 1)[:, None, None]
        u = np.linspace(0, 1, P)
        w = (smoothstep(start, 1.0, u) ** power)[None, :] * k_tip[:, None]
        # a strand's own offset at the root is kept (the clump narrows toward the tips)
        return X + w[..., None] * (C[ids] - X)

    def shape(self, X, X0, lab):
        H, rng, k = self.H, self.rng, self.k
        S, P, _ = X.shape
        T, B, Nn = self.frame_along(X)
        # big locks: coherent waves
        big, nb = self.clump_ids(X0, lab, 85)
        seg = np.linalg.norm(np.diff(X, axis=1), axis=2)
        s = np.concatenate([np.zeros((S, 1)), np.cumsum(seg, 1)], 1)
        wave_on = smoothstep(H.eye_y + STYLE['wave_top'] * H.u, H.eye_y - 0.15 * self.d, X[..., 1])
        if STYLE.get('coherent', False):
            # HAIR2 (6): waves coherent across neighbouring locks — phase is a smooth field of height + root azimuth
            # (+ a small per-lock / per-strand jitter), so the highlights line up into broken horizontal bands instead
            # of lock-sized blobs
            azr = np.radians(H.az_el(X0)[0])
            fld = 1.1 * np.sin(1.3 * azr + 0.4) + 0.5 * np.sin(3.1 * azr + 1.1)
            lam0 = (STYLE['lam'][0] + 0.5 * STYLE['lam'][1]) * k * (1.0 + 0.06 * np.sin(2.3 * azr + 0.9))
            ph_l = rng.normal(0.0, 0.35, nb)
            arg = 2 * np.pi * (H.eye_y - X[..., 1]) / lam0[:, None] + (fld + ph_l[big] + rng.normal(0, 0.12, S))[:, None]
            amp_l = 1.0 + 0.2 * rng.standard_normal(nb)
            A = (STYLE['amp'][0] + 0.5 * STYLE['amp'][1]) * k * amp_l[big][:, None] * wave_on * (0.9 + 0.2 * rng.random(S))[:, None]
        else:
            ph = rng.random(nb) * 2 * np.pi
            lam = (STYLE['lam'][0] + STYLE['lam'][1] * rng.random(nb)) * k
            amp = (STYLE['amp'][0] + STYLE['amp'][1] * rng.random(nb)) * k
            jitter = rng.random(S) * 0.6
            arg = 2 * np.pi * s / lam[big][:, None] + (ph[big] + jitter)[:, None]
            A = amp[big][:, None] * wave_on * (0.75 + 0.5 * rng.random(S))[:, None]
        X = X + (A * np.sin(arg))[..., None] * B + (0.3 * A * np.cos(arg))[..., None] * Nn
        # lob silhouette: a soft outward bulge at the jaw line, the ends curl slightly in toward the neck
        u_ = np.linspace(0, 1, P)[None, :]
        hz = X - H.O; hz[..., 1] = 0.0; hz = nrm(hz)
        hang = smoothstep(H.ear["l"]["bot"], H.chin[1], X[..., 1])          # 0 above the ear lobe .. 1 below the chin
        bulge = STYLE['bulge'] * k * np.sin(np.pi * np.clip(hang, 0, 1))
        curl = -STYLE['curl'] * k * smoothstep(0.72, 1.0, u_) ** 2 * (hang > 0.3)
        if STYLE.get('coherent', False):
            # HAIR2 (7): no dead-straight curtain edge — per-lock bulge and lock ends that stand off / tuck in
            bl = np.clip(1.0 + 0.45 * rng.standard_normal(nb), 0.2, 2.2)[big][:, None]
            kick = (rng.normal(0.0, 0.0022, nb) + (rng.random(nb) < 0.12) * rng.uniform(0.003, 0.008, nb)) * k
            bulge = bulge * bl + kick[big][:, None] * smoothstep(0.55, 1.0, u_) * (hang > 0.2)
        X = X + ((bulge + curl) * (u_ > 0.2))[..., None] * hz
        # 5-8 distinct face-framing locks at the front of both sides (tight, a touch forward)
        az0, el0, _ = H.az_el(X0)
        dyf = self.density(X0)[1]
        fl = np.zeros(nb, bool)
        cand = np.unique(big[(np.abs(az0) > 38) & (np.abs(az0) < 72) & (dyf < 0.02 * k)])
        if len(cand):
            fl[rng.choice(cand, min(7, len(cand)), replace=False)] = True
        self.face_lock = fl[big]
        X = X + ((0.004 * k * smoothstep(0.3, 0.8, u_)) * self.face_lock[:, None])[..., None] * np.array([0.0, 0.0, 1.0])
        # clumps at 3 scales
        kb = 0.50 + 0.25 * rng.random(S)
        kb = np.where(self.face_lock, 0.85, kb)
        X = self.pull(X, big, nb, kb, start=0.25)
        # lock layering: each big lock rides a little higher/lower over its neighbours (overlap + self-shadow
        # between locks instead of one smooth sheet)
        T2, B2, N2 = self.frame_along(X)
        off = ((rng.random(nb) - 0.35) * 0.005 * k)[big]
        u_ = np.linspace(0, 1, P)[None, :]
        X = X + (off[:, None] * smoothstep(0.05, 0.4, u_))[..., None] * N2
        if getattr(self, "mid_pre", None) is not None:
            mid, nm = self.mid_pre, int(self.mid_pre.max()) + 1
        else:
            mid, nm = self.clump_ids(X0, lab, 1400)
        X = self.pull(X, mid, nm, 0.80 + 0.15 * rng.random(S), start=0.08, power=0.6)
        small, ns = self.clump_ids(X0, lab, 9000)
        X = self.pull(X, small, ns, 0.70 + 0.25 * rng.random(S), start=0.04, power=0.5)
        # frizz: smooth per-strand noise growing to the tips; a few strays
        u = np.linspace(0, 1, P)[None, :]
        fr = np.zeros_like(X)
        for f, a0 in ((2.0, 0.0006), (5.0, 0.00035)):
            phs = rng.random((S, 3)) * 2 * np.pi
            fr += a0 * k * np.sin(2 * np.pi * f * u[..., None] + phs[:, None, :])
        stray = rng.random(S) < 0.006
        fr[stray] *= 6.0
        X = X + fr * smoothstep(0.1, 1.0, u)[..., None]
        # collision again (head + neck / clothes) and roots pinned
        Xr = X[:, :1].copy()
        X = H.push(X, 0.0012 * k, iters=2, body=0.014 * k)
        X[:, :1] = Xr
        self.mid = mid
        return X, small

    # ------------------------------------------------------------------ 5 fuzz
    def fuzz(self, n_baby, n_fly, scalpX, lab_s, outer=None, n_cross=0):
        H, rng, k = self.H, self.rng, self.k
        # baby hairs: roots in the hairline band, short, finer, curling along the comb direction
        H_ = self.H
        cand, cn = H_.sample(n_baby * 120, rng)
        dens, dy = self.density(cand)
        X0, N0 = cand, cn
        band = (dy < 0.007 * k) & (dy > -0.004 * k) & (rng.random(len(dy)) < 0.15 + dens)
        X0, N0, dy = X0[band][:n_baby], N0[band][:n_baby], dy[band][:n_baby]
        lab = self.labels(X0)
        t0 = self.comb(X0, N0, lab)
        S = len(X0)
        L = np.where(rng.random(S) < 0.7, 0.005 + 0.010 * rng.random(S), 0.015 + 0.025 * rng.random(S)) * k   # baby hairs + wisps
        u = np.linspace(0, 1, P_FUZZ)
        side = nrm(np.cross(N0, t0))
        curl = (rng.random(S) - 0.5) * 2.5
        lift = 0.15 + 0.4 * rng.random(S)
        ang = curl[:, None] * u[None, :]
        Dt = np.cos(ang)[..., None] * t0[:, None, :] + np.sin(ang)[..., None] * side[:, None, :]
        Dt = nrm(Dt + (lift[:, None] * (1 - u[None, :]))[..., None] * N0[:, None, :])
        Xb = X0[:, None, :] + np.cumsum(Dt * (L / (P_FUZZ - 1))[:, None, None], axis=1) - Dt[:, :1] * (L / (P_FUZZ - 1))[:, None, None]
        Xb = H.push(Xb, 0.0004 * k, iters=2)
        Xb[:, 0] = X0
        wb = np.stack([np.full(S, 2.0e-5), np.full(S, 0.4e-5)], 1)
        # flyaways: leave the outer layer of a few strands mid-length and arc out (tapered, few)
        Sx = len(scalpX)
        pool = np.where(outer)[0] if outer is not None and outer.sum() > n_fly + n_cross else np.arange(Sx)
        pick = rng.choice(pool, n_fly, replace=False)
        base = scalpX[pick]
        P = base.shape[1]
        j0 = rng.integers(P // 6, (3 * P) // 4, n_fly)
        Xf = np.zeros((n_fly, P_FUZZ, 3))
        for i in range(n_fly):
            seg = base[i, j0[i]:]
            Xf[i] = resample(seg[None], P_FUZZ)[0]
        Ts, Bs, Ns = self.frame_along(Xf)
        uu = np.linspace(0, 1, P_FUZZ)[None, :, None]
        # most arc a few mm off the mass, a third leave it by 1.5-3.5 cm (they catch the rim light off the silhouette)
        dev = np.where(rng.random(n_fly) < 0.33, 0.015 + 0.020 * rng.random(n_fly), 0.003 + 0.010 * rng.random(n_fly))
        dev = dev[:, None, None] * k
        sgn = np.where(rng.random(n_fly) < 0.5, -1, 1)[:, None, None]
        out = nrm(Xf - H.O)
        Xf = Xf + dev * uu ** 1.6 * (0.8 * out + 0.5 * sgn * Bs + 0.3 * (rng.random((n_fly, 1, 3)) - 0.5))
        keepf = rng.random(n_fly) < 1.0
        Xf = Xf[keepf]
        # flyaway "roots" are mid-strand: the renderer binds by the first point, so keep it at the base strand
        wf = np.stack([np.full(len(Xf), 3.0e-5), np.full(len(Xf), 0.6e-5)], 1)
        parts, Ws = [Xb, Xf], [wb, wf]
        # strays: segments of outer strands lying ON the mass but turned 15-45 deg off the flow (cross over locks)
        if n_cross:
            pc = rng.choice(pool, n_cross, replace=False)
            bc = scalpX[pc]
            j0 = rng.integers(P // 8, P // 2, n_cross)
            ln = rng.integers(P // 5, P // 2, n_cross)
            Xc = np.zeros((n_cross, P_FUZZ, 3))
            for i in range(n_cross):
                Xc[i] = resample(bc[i, j0[i]:min(P, j0[i] + ln[i])][None], P_FUZZ)[0]
            nrm_o = nrm(Xc[:, :1] - H.O)
            ang = np.radians(rng.uniform(15, 45, n_cross) * np.where(rng.random(n_cross) < 0.5, -1, 1))
            r0 = Xc[:, :1]
            v = Xc - r0
            c, s_ = np.cos(ang)[:, None, None], np.sin(ang)[:, None, None]
            v = v * c + np.cross(nrm_o, v) * s_ + nrm_o * (nrm_o * v).sum(-1, keepdims=True) * (1 - c)   # Rodrigues
            uu = np.linspace(0, 1, P_FUZZ)[None, :, None]
            Xc = r0 + v + nrm_o * (0.0008 + 0.0012 * rng.random((n_cross, 1, 1))) * k * np.sin(np.pi * uu) ** 0.5
            Xc = H.push(Xc, 0.002 * k, iters=1)
            parts.append(Xc); Ws.append(np.stack([np.full(n_cross, 4.0e-5), np.full(n_cross, 1.0e-5)], 1))
        Xall = np.concatenate(parts, 0)
        W = np.concatenate(Ws, 0)
        return Xall, W


INTERIOR_MM, INTERIOR_KEEP = 2.5, 0.25
N_REF = 120000      # widths are set for this density; fewer strands get wider (same mass / coverage)


def shell_depth(H, X, nb=(24, 48)):
    """per strand: min depth (m) below the groom's outer shell over the strand's points past 15 % of its length.
    Shell = the 92nd-percentile radius per direction bin: spherical about the skull centre above the ear line,
    cylindrical (about the vertical axis through it, per height slab) below."""
    S, P, _ = X.shape
    pts = X.reshape(-1, 3)
    d = pts - H.O
    lowp = d[:, 1] < -0.3 * H.R
    az = np.arctan2(d[:, 0], d[:, 2])
    ia = np.clip(((az + np.pi) / (2 * np.pi) * nb[1]).astype(int), 0, nb[1] - 1)
    r_s = np.linalg.norm(d, axis=1)
    el = np.arcsin(np.clip(d[:, 1] / np.maximum(r_s, 1e-9), -1, 1))
    ie = np.clip(((el + np.pi / 2) / np.pi * nb[0]).astype(int), 0, nb[0] - 1)
    r_c = np.hypot(d[:, 0], d[:, 2])
    iy = np.clip(((d[:, 1] + 0.3 * H.R) / -0.008).astype(int), 0, 60)
    b = np.where(lowp, 100000 + iy * nb[1] + ia, ie * nb[1] + ia)
    r = np.where(lowp, r_c, r_s)
    ub, inv = np.unique(b, return_inverse=True)
    o = np.lexsort((r, inv)); rs = r[o]; bs = inv[o]
    cnt = np.bincount(inv, minlength=len(ub)); st = np.concatenate([[0], np.cumsum(cnt)[:-1]])
    shell = rs[st + (0.92 * (cnt - 1)).astype(int)]
    dep = (shell[inv] - r).reshape(S, P)
    return dep[:, int(P * 0.15):].min(1)


def _arrays(X, W, clump, rng):
    return {"pos": X.astype(np.float32), "width": W.astype(np.float32),
            "seed": rng.random(len(X)).astype(np.float32), "clump": np.asarray(clump, np.uint32)}


def build(H, rng, n=120000, n1=26000, n_baby=2400, n_fly=750, n_cross=350, log=print):
    import time
    t0 = time.time()
    g = Groom(H, rng)
    X0, N0, dy, dens = g.roots(n)
    lab = g.labels(X0)
    log(f"[scalp] roots {len(X0)} (big {np.sum(lab == 0)}, small {np.sum(lab == 1)}, back {np.sum(lab == 2)}) "
        f"{time.time() - t0:.1f}s")
    g.mid_pre, _ = g.clump_ids(X0, lab, 1100)
    Xs, y_end = g.march(X0, N0, lab, dy)
    log(f"[scalp] march {Xs.shape} {time.time() - t0:.1f}s; length mm p5/50/95 "
        f"{np.percentile(g.length, [5, 50, 95]).round(3) * 1000}")
    # smooth the marched polylines (the shell projection leaves a 1-2 mm zig-zag at the step scale), root pinned
    for _ in range(4):
        Xs[:, 1:-1] = 0.5 * Xs[:, 1:-1] + 0.25 * (Xs[:, :-2] + Xs[:, 2:])
    Xs = H.push(Xs, 0.0015 * g.k, iters=1)
    Xs[:, 0] = X0
    X = resample(Xs, P_SCALP)
    X, small = g.shape(X, X0, lab)
    log(f"[scalp] shaped {time.time() - t0:.1f}s")
    S = len(X)
    # interior thinning (fill budget): strands that stay >= INTERIOR_MM under the groom's outer shell over their
    # whole visible length are never seen; keep a fraction of them (wider, so the mass still occludes / shadows)
    dep = shell_depth(H, X)
    inner = dep > INTERIOR_MM * 1e-3 * g.k
    keep = ~inner | (rng.random(S) < INTERIOR_KEEP)
    wk = np.where(inner, 1.0 / np.sqrt(INTERIOR_KEEP), 1.0)[keep]
    log(f"[scalp] interior strands {inner.mean() * 100:.0f}% -> kept {keep.sum()} of {S}")
    X, small, dy, X0 = X[keep], small[keep], dy[keep], X0[keep]
    g.mid = g.mid[keep]
    S = len(X)
    # width: human hair ~70 um at the root, tapering; hairline strands finer
    fine = smoothstep(-0.002 * g.k, 0.012 * g.k, dy)
    wr = (4.5e-5 + 3.0e-5 * fine) * (0.85 + 0.3 * rng.random(S)) * (1.0 + 0.5 * (1.0 - STYLE.get('fall', 0.0)) * g.near_part[keep])
    wr = wr * wk * np.sqrt(max(1.0, N_REF / max(n, 1)))
    W = np.stack([wr, wr * 0.30], 1)
    sub = rng.choice(S, min(n1, S), replace=False)
    meta = {"bone": "head", "follow": "bone", "surface_submesh": "head", "width_root_m": float(np.median(wr)),
            "width_tip_m": float(np.median(wr) * 0.22), "color_root": HAIR_ROOT, "color_tip": HAIR_TIP,
            "melanin": 0.72, "redness": 0.35, "roughness": 0.30, "groom": "junor lob: side part, soft waves, left tuck",
            "clumps": int(small.max() + 1)}
    k1 = S / len(sub)                # linear: sub-pixel strands keep the mass coverage only if width x count is kept
    W1 = W[sub] * k1
    cid = g.mid                     # lock-level ids: the renderer hashes them into per-lock shade / shift noise
    scalp = {"meta": meta, "lod0": _arrays(X, W, cid, rng), "lod1": _arrays(X[sub], W1, cid[sub], rng)}
    outer = dep[keep] < 0.8e-3 * g.k
    Xf, Wf = g.fuzz(n_baby, n_fly, X, lab, outer=outer, n_cross=n_cross)
    subf = rng.choice(len(Xf), len(Xf) // 3, replace=False)
    fmeta = dict(meta, groom="baby hairs + flyaways", width_root_m=3.2e-5, width_tip_m=0.8e-5)
    fmeta.pop("clumps", None)
    fuzz = {"meta": fmeta, "lod0": _arrays(Xf, Wf, np.arange(len(Xf)), rng),
            "lod1": _arrays(Xf[subf], Wf[subf] * np.sqrt(3), subf, rng)}
    log(f"[scalp] done: scalp {S} x {P_SCALP}, lod1 {len(sub)}, fuzz {len(Xf)} {time.time() - t0:.1f}s")
    return {"scalp": scalp, "fuzz": fuzz}
