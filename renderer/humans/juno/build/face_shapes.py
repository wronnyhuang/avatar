"""asset — Juno face blendshapes on the MH skin vertices (owner: face-shapes helper). Procedural rig (face_rig.py).

build(base) -> {name: (nV,3) float64 dpos in metres} for EVERY name in asset_io.REQUIRED_SHAPES + correctives corr_*.
Deltas are zero outside the skin ("body" group). Everything is measured from the base (landmarks, lid margins from
juno_sculpt.lid_margins, MH topology/targets used as weight fields only), so the lead's design changes carry through.

Construction
  jawOpen      exact rigid rotation of the skin about base.lm["jaw_hinge"] around the jaw bone's local +x by
               JAW_OPEN_DEG * w_jaw (w_jaw = MH mouth-open falloff: chin/lower lip 1 -> upper lip/skull 0). Linear in
               the weight (no arc corrective) so lip sealing below is exact. jaw_open_deg() returns JAW_OPEN_DEG with
               the sign the jaw bone needs.
  lips         seal field c (1 on both lips near the inner-lip seam incl. corners, 0 at chin/cheeks):
               mouthClose (+ corr_jawOpen_mouthClose) seals the seam (ARKit: mouthClose <= jawOpen; lips meet when equal);
               lipsTogether / mouthPress* seal at ANY jaw weight via the product correctives below.
  lids         rotations of the lid shells about the eyeball centre (base.eye[side]) around +x, radius riding the
               cornea profile (face_rig.lid_floor = eyes_mouth cornea + clearance). Big rotations (eyeBlink) are
               split w*A + w^2*B (corr_eyeBlinkX_eyeBlinkX = B) so the path follows the arc at every weight.

Correctives (rule = product of the two drivers; a self-product = the driver squared). CORRECTIVES maps name -> drivers.
  corr_eyeBlinkL_eyeBlinkL        arc in-between of the blink (w^2 term)                               [L/R alike]
  corr_eyeBlinkL_<X>              X in eyeSquint, cheekSquint, eyeHappy, eyeWide, eyeLookUp, eyeLookDown, mouthSmile:
                                  the closed upper lid lands on X's lower lid and cancels X's own upper-lid move
  corr_jawOpen_jawOpen            sticky lips (r3): jawOpen = J - STICK*c*J, this = +STICK*c*J, so the lip seam opens as
                                  (1-STICK) j + STICK j^2 (lips stay together on a barely open jaw, exact at j=1; the
                                  skin off the lips and the jaw bone stay the exact linear rotation)
  corr_jawOpen_mouthClose         mouthClose = -(1-STICK)*c*J, this = -STICK*c*J: lips meet exactly at mouthClose = jawOpen
  corr_jawOpen_lipsTogether       -PRESS_SEAL*c*J: lipsTogether=1 seals exactly at jawOpen .3 (<= 0.6 mm overlap below,
                                  gap STICK*j*(j-.3) above)
  corr_jawOpen_mouthPressL/R      the same per side
  corr_jawOpen_mouthSmileL/R      SMILE_SEAL*(-c*J): a smile on a slightly open jaw keeps its lips closed
  corr_jawOpen_mouthPucker        pucker on an open jaw keeps a rounded, forward lip ring
"""
from __future__ import annotations

import numpy as np

from face_rig import FaceRig, rotate, gauss
from juno_base import smoothstep

JAW_OPEN_DEG = 22.0
STICK = 0.6          # mouth3 (was .8: speech at jawOpen .3-.45 barely parted the lips) sticky-lip share of the seam opening that grows as jawOpen^2
GUARD = 0.0030       # front skin stays this far in front of the teeth-arch lining fit
TEETH_GUARD = 0.0026  # smile2: ... and this far in front of the real teeth/gum surface (lod0 subdivision shrinks ~0.5 mm)
REST_GUARD = 0.0008   # mouth4: rest allowance reference (see guard)
PRESS_SEAL = (1 - STICK) + STICK * 0.3   # press/lipsTogether seal exactly at jawOpen .3 (tiny overlap below, small gap above)
SMILE_SEAL = -0.55   # mouth3 (was -.30: broad open smile shows the upper teeth row) lip seal per (jawOpen x mouthSmile) (smile2: < 0 = an open smile parts the lips faster -> upper teeth)
# smile2 ( asset-smile2) amplitudes, mouth half-width (mw) units
SM_UP, SM_OUT, SM_BACK = 0.42, 0.21, 0.10     # corner (pr = 1) up / out / back (mouth4: was .36/.19, corners up AND wide)
SM_THIN_LO, SM_LO_W = 0.85, 0.45              # lower lip thins: lower border rises SM_THIN_LO * g(h) toward the seam
SM_LO_FALL = 0.45                             # the smile field below the lower lip falls off over this (chin stays)
SM_THIN_UP, SM_UP_W = 0.25, 0.45              # upper lip thins toward the seam
SM_FOLD, SM_POCKET = 0.12, 0.06               # nasolabial bulge forward / pocket beside the corner (back)
SMILE_GUARD_J = (0.15, 0.3)                   # jawOpen values the open-smile combination guard fixes
FV_UP_LIFT = 0.20                             # labioDental upper lip lift (was 0.07: FV read as closed lips)
JAW_LIP_ROLL = 14.0                           # deg the lower lip rolls in at jawOpen 1 (no everted lining)
FUNNEL_NARROW, FUNNEL_OPEN_UP, FUNNEL_OPEN_LO = 0.45, 0.20, 0.22   # smile2: OH reads rounder (was .32/.10/.12)
SM_TEETH_LIFT = 0.45                          # (jawOpen x smile) upper lip centre lift off the incisors (mouth4: was .32)
SM_APPLE_UP, SM_APPLE_FWD, SM_LID = 0.10, 0.08, 0.11   # mouth4: apples .085/.07, lid .10 -> fuller acted smile
SM_LO_DROP = 0.80                             # mouth4: (jawOpen x smile) lower lip centre drops off the upper teeth (D-shaped open smile)
UPPER_UP = 0.30                               # mouth4: mouthUpperUp lift (was .24 mw)
SIDES = (("Left", "l", 1.0), ("Right", "r", -1.0))

CORRECTIVES = {}


def _jaw_sign(rig):
    return 1.0 if rig.a_ref > 0 else -1.0


def build(base):
    rig = FaceRig(base)
    VA, nV, skin = base.VA, base.nV, base.skin_mask
    skf = skin.astype(float)
    mw, iod = rig.mw, rig.iod
    x = VA[:, 0]
    u, h, dz = rig.u, rig.h, rig.dz
    S = {}
    CORRECTIVES.clear()

    def V3(*a):
        return np.array(a, float)

    # teeth guard: front skin never ends behind the procedural teeth arch (mouth_teeth lining fit) + GUARD
    import eyes_mouth
    import mouth_teeth
    cav = eyes_mouth._cavity_faces(base)
    cv = np.unique(np.concatenate([np.array([v for v, _ in base.obj.faces[f]]) for f in cav]))
    arch_u, arch_l = mouth_teeth.arch_curves(base, rig, cv)
    sy0 = float(np.interp(0.0, rig.seam_x, rig.seam_y))
    yw = smoothstep(sy0 - 0.12 * mw, sy0 + 0.04 * mw, VA[:, 1])        # 1 = upper arch, 0 = lower (rest height)
    zg_v = lambda xx, w_: w_ * arch_u(xx) + (1 - w_) * arch_l(xx) + GUARD
    xend = 1.5 * mw
    gband = skin.copy()
    gband[cv] = False
    gband &= (np.abs(x) < xend) & (VA[:, 1] > sy0 - 0.8 * mw) & (VA[:, 1] < sy0 + 0.55 * mw) & (VA[:, 2] >= zg_v(x, yw))
    gi = np.where(gband)[0]

    # smile2: + a guard on the real teeth/gum geometry (front-most z per (x, y), 0.5 mm grid, 2.5 mm max filter): the
    # arch guard above has no height profile, so skin lifted to the gum line (a thinning lower lip) ended ~0.9 mm in
    # front of the gum bead on the subdivided lod0 skin and the gums showed through in real-time (dark marks)
    from scipy.ndimage import maximum_filter
    TG = mouth_teeth.build(base, rig, cv)[0]
    th_j = np.radians(JAW_OPEN_DEG) * _jaw_sign(rig)
    GS = 0.0005
    gx0, gy0 = -1.8 * mw, sy0 - 1.8 * mw
    gnx, gny = int(3.6 * mw / GS) + 1, int(2.8 * mw / GS) + 1

    def front_map(jt=0.0):
        Q = np.concatenate([rotate(p.P, rig.hinge, rig.jaw_axis, np.full(len(p.P), th_j * jt))
                            if (jt and p.J[0, 0] == base.bidx["jaw"]) else p.P for p in TG])
        Z = np.full((gny, gnx), -1.0)
        ix = np.clip(((Q[:, 0] - gx0) / GS).astype(int), 0, gnx - 1)
        iy = np.clip(((Q[:, 1] - gy0) / GS).astype(int), 0, gny - 1)
        np.maximum.at(Z, (iy, ix), Q[:, 2])
        Z = maximum_filter(Z, size=5)

        def zf(xx, yy):
            jx = np.clip(((xx - gx0) / GS).astype(int), 0, gnx - 1)
            jy = np.clip(((yy - gy0) / GS).astype(int), 0, gny - 1)
            return Z[jy, jx]
        return zf
    zf0 = front_map(0.0)
    # smile2: also the labiomental / chin-front skin in front of the real gums (it sits behind the lining-fit arch
    # + GUARD at rest and was outside the band, so nothing stopped it from being lifted into the gum line)
    gband |= (skin & ~np.isin(np.arange(nV), cv) & (np.abs(x) < xend) & (VA[:, 1] > sy0 - 1.2 * mw)
              & (VA[:, 1] < sy0 + 0.55 * mw) & (dz > -0.9) & (VA[:, 2] >= zf0(VA[:, 0], VA[:, 1]) - 0.0005))   # mouth4: was +.5 mm
    gi = np.where(gband)[0]

    def guard(D, P0=None, zf=None):
        k = gi[np.abs(D[gi]).sum(1) > 1e-7]
        if not len(k):
            return D
        P0k = VA[k] if P0 is None else P0[k]
        zf = zf0 if zf is None else zf
        P1 = P0k + D[k]
        z1 = np.maximum(zg_v(np.clip(P1[:, 0], -xend, xend), yw[k]), zf(P1[:, 0], P1[:, 1]) + TEETH_GUARD)
        # mouth4: the rest allowance is measured against the REAL teeth surface (+ REST_GUARD), not the full margin: chin
        # skin that sits 0.5-2 mm in front of the lower gums at rest and is lifted into the gum line by the (wider) smile
        # kept its whole rest deficit and the gums came through below the lip
        z0 = np.maximum(zg_v(np.clip(P0k[:, 0], -xend, xend), yw[k]) - GUARD + REST_GUARD, zf(P0k[:, 0], P0k[:, 1]) + REST_GUARD)
        d = P1[:, 2] - z1
        kk = 0.0008
        fix = 0.5 * (d + np.sqrt(d * d + kk * kk)) - d            # smooth max(d, 0) - d  (>= 0, ~0 when well in front)
        d0 = P0k[:, 2] - z0
        fix = np.maximum(fix - (0.5 * (d0 + np.sqrt(d0 * d0 + kk * kk)) - d0), 0)   # minus the rest offset of the soft max
        fix *= 1 - smoothstep(0.85 * xend, xend, np.abs(P1[:, 0]))
        F = np.zeros(nV)
        F[k] = fix
        F = np.maximum(F, rig.smooth(F, 2))
        F[F < 2e-5] = 0.0
        D = D.copy()
        D[:, 2] += F * gband
        return D

    def put(name, D, seal=True):
        D = np.asarray(D, float) * skf[:, None]
        D = guard(D)
        S[name] = rig.seal(D) if seal else D

    def corr(a, b, D, seal=False):
        nm = f"corr_{a}_{b}"
        CORRECTIVES[nm] = [a, b]
        S[nm] = np.asarray(D, float) * skf[:, None]

    sideW = {"l": smoothstep(-0.18 * mw, 0.18 * mw, x)}
    sideW["r"] = 1 - sideW["l"]
    sideW0 = sideW
    front = smoothstep(-1.4, -0.6, dz)                    # in front of the mouth bag (mw units)

    # lip rolls: rotate the vermilion about an axis behind the lip line
    def roll(which, deg):
        sgn = 1 if which == "upper" else -1
        m = (rig.upper if which == "upper" else rig.lower).astype(float)
        hh = h if which == "upper" else -h
        wv = m * np.exp(-(np.maximum(hh, 0) / 0.42) ** 2) * (1 - smoothstep(0.9, 1.35, np.abs(u))) * front * (1 - smoothstep(0.7, 1.4, -hh))   # r3: not the chest/neck below
        wv = np.clip(rig.smooth(wv, 1), 0, 1) * skf
        piv = np.c_[VA[:, 0], np.interp(np.clip(x, -mw, mw), rig.seam_x, rig.seam_y) + sgn * 0.12 * mw,
                    np.interp(np.clip(x, -mw, mw), rig.seam_x, rig.seam_z) - 0.30 * mw]
        q = VA - piv
        a = np.radians(deg) * sgn * wv
        ca, sa = np.cos(a), np.sin(a)
        Dr = np.zeros((nV, 3))
        Dr[:, 1] = q[:, 1] * ca - q[:, 2] * sa - q[:, 1]
        Dr[:, 2] = q[:, 1] * sa + q[:, 2] * ca - q[:, 2]
        return Dr * skf[:, None]

    # ------------------------------------------------------------------ jaw
    th = np.radians(JAW_OPEN_DEG) * _jaw_sign(rig)
    J = (rotate(VA, rig.hinge, rig.jaw_axis, th * rig.w_jaw) - VA) * skf[:, None]
    front0 = smoothstep(-1.4, -0.6, dz)
    # oval opening: near the lip line the lower lip follows the jaw by sqrt(1-u^2) (corners stay closed, the middle
    # drops fully); chin/jaw below the lip keep the exact rotation
    prof = np.clip(1 - (np.abs(u) / 1.06) ** 2.6, 0, 1) ** 0.42      # tall soft oval: flatter bottom than a circle
    near_lip = np.exp(-(np.minimum(h + 0.15, 0) / 0.55) ** 2) * (1 - smoothstep(1.0, 1.6, np.abs(u))) * front0
    near_lip = np.clip(rig.smooth(near_lip * rig.lower, 2), 0, 1) * skf
    wj = rig.w_jaw * (1 - near_lip * (1 - prof) ** 1.3)
    J = (rotate(VA, rig.hinge, rig.jaw_axis, th * wj) - VA) * skf[:, None]
    # upper lip: centre lifts off the teeth (so the upper incisors show), vermilion rolls in a little
    ul = rig.upper * np.exp(-(np.maximum(h, 0) / 0.75) ** 2) * prof ** 0.8 * front0 * skf
    ul = np.clip(rig.smooth(ul, 2), 0, 1) * skf
    LIFT = 0.10 * mw
    Jx = ul[:, None] * V3(0, LIFT, -0.30 * LIFT)
    # corners: drawn slightly in and down (the oval narrows), lips stretched thin over the teeth
    for s_, sg_ in (("l", 1.0), ("r", -1.0)):
        cg = gauss(VA, rig.corner[s_], 0.32 * mw) * sideW0[s_] * front0 * skf
        Jx += cg[:, None] * V3(-sg_ * 0.07, -0.05, -0.03) * mw
    # inner-lip roll: the lips thin and turn in over the teeth as the mouth opens (mouthClose undoes it on the seam)
    # lower lip: thins and lies back against the lower teeth (taut, no puff) - a translation, not a roll
    lw2 = rig.lower * np.exp(-(np.minimum(h, 0) / 0.55) ** 2) * prof * front0 * skf
    lw2 = np.clip(rig.smooth(lw2, 2), 0, 1) * skf
    Jx += lw2[:, None] * V3(0, -0.04, -0.02) * mw
    J = J + Jx
    # smile2: the open jaw everted the lower lip (its lining faced the camera as a bright orange crescent below the
    # mouth in real-time at jawOpen 1): roll the lower vermilion in over the teeth as the jaw opens
    J = J + roll("lower", JAW_LIP_ROLL)
    S["jawOpen"] = J
    up_ax = V3(0, 1, 0)
    S["jawLeft"] = (rotate(VA, rig.hinge, up_ax, np.radians(4.0) * rig.w_jaw) - VA) * skf[:, None]
    S["jawRight"] = (rotate(VA, rig.hinge, up_ax, -np.radians(4.0) * rig.w_jaw) - VA) * skf[:, None]
    S["jawForward"] = rig.w_jaw[:, None] * V3(0, 0, 0.003) * skf[:, None]

    # ------------------------------------------------------------------ lip seal field
    c_seal = ((1 - smoothstep(0.30, 0.95, -h)) * (1 - smoothstep(1.15, 1.7, np.abs(u))) * front) * skf
    c_seal = np.clip(rig.smooth(c_seal, 1), 0, 1)
    c_seal[rig.seam_u] = 1.0
    c_seal[rig.seam_l] = 1.0
    seal = -J * c_seal[:, None]
    # sticky lips: the lip seam opens as (1-STICK) j + STICK j^2 (closed-ish at small jaw weights, exact at 1);
    # mouthClose is split the same way so jawOpen = mouthClose = j seals exactly at every j
    S["jawOpen"] = J + STICK * seal
    corr("jawOpen", "jawOpen", -STICK * seal)
    S["mouthClose"] = (1 - STICK) * seal
    corr("jawOpen", "mouthClose", STICK * seal)

    # mouth region fields
    mo = rig.mouth_c
    lipband = np.exp(-(h / 0.42) ** 2) * np.exp(-(np.maximum(np.abs(u) - 0.9, 0) / 0.35) ** 2) * front * skf
    mouthR = np.exp(-((VA - mo) ** 2).sum(1) / (2 * (1.05 * mw) ** 2)) * front * skf

    # press: lips compress toward the lip line (seam vertices fixed -> no gap), flatten a touch
    press = np.zeros((nV, 3))
    comp = np.exp(-(h / 0.55) ** 2) * (1 - smoothstep(1.0, 1.4, np.abs(u))) * front * skf
    press[:, 1] = -0.30 * h * mw * comp
    press[:, 2] = -0.05 * mw * comp * np.exp(-(h / 0.3) ** 2)
    for side, s, sg in SIDES:
        put(f"mouthPress{side}", press * sideW[s][:, None], seal=False)
        corr("jawOpen", f"mouthPress{side}", PRESS_SEAL * seal * sideW[s][:, None])
    put("lipsTogether", 0.5 * press, seal=False)
    corr("jawOpen", "lipsTogether", PRESS_SEAL * seal)

    # ------------------------------------------------------------------ corner-driven shapes
    def corner_field(s, sx=0.55, sy=0.45, lat=0.15):
        cc = rig.corner[s] + V3((1 if s == "l" else -1) * lat * mw, 0, 0)
        g = np.exp(-(((VA[:, 0] - cc[0]) / (sx * mw)) ** 2 + ((VA[:, 1] - cc[1]) / (sy * mw)) ** 2) / 2)
        # lips toward the centre follow less; nothing crosses the midline
        return g * sideW[s] * smoothstep(-1.6, -0.5, dz) * skf

    cheek_gate = {}
    for side, s, sg in SIDES:
        E = rig.eyes[s]
        cheek_gate[s] = smoothstep(1.08 * E["R"], 1.9 * E["R"], E["r"])

    def cheek_lift(s, amt):
        E = rig.eyes[s]
        R = E["R"]
        cc = rig.cheek[s]
        g = gauss(VA, cc, 0.36 * iod) * cheek_gate[s] * sideW[s] * skf
        D = g[:, None] * (V3((1 if s == "l" else -1) * 0.012, 0.045, 0.018) * iod * amt)
        return D

    def lower_lid(s, frac, arch=0.0):
        """lower lid up by frac of the aperture (frac<0 = down); arch>0 lifts the centre more (^_^)."""
        E = rig.eyes[s]
        ap = rig.aperture(s)
        a0, a1 = E["azb"][0], E["azb"][-1]

        def f(az):
            t = np.clip((az - 0.5 * (a0 + a1)) / (0.5 * (a1 - a0)), -1, 1)
            prof = (1 - arch) + arch * (1 - t * t) ** 0.7 * 1.25
            return -frac * ap(az) * prof
        return rig.lid_delta(s, "lo", f), f

    def upper_lid(s, frac=None, deg=None, path=False):
        ap = rig.aperture(s)
        if deg is not None:
            f = lambda az: np.full_like(az, np.radians(deg))
        else:
            f = lambda az: frac * ap(az)
        return rig.lid_delta(s, "up", f, path=path), f

    # per-shape lower/upper lid motion (functions of az, + = down) for the blink correctives
    LID = {}

    for side, s, sg in SIDES:
        sx = sg
        R = rig.eyes[s]["R"]
        cf = corner_field(s)
        # smile: one smooth field over both lips (no shear across the lip -> closed, no curl): the lip line bends up
        # toward this corner (t^2 profile), corner goes up, back along the cheek and out; the lips flatten onto the
        # teeth and thin (no forward push); the cheek apple lifts and comes forward, the nasolabial fold forms (the
        # skin lateral of the alar->corner line rides forward/up, medial side stays), lower lid rises a little
        # smile2 ( asset-smile2): the lip LINE bends up toward the corner (pr, t^2 profile) while both lips THIN
        # toward the seam (the lower lip's lower border rises more than the seam -> no U-bowl pout), the lips stretch
        # wide and flat over the teeth, the corner goes up/out and only a little back (stays visible from the front)
        # with a small pocket beside it; cheek apple lifts and rounds, soft nasolabial bulge, lower lid up a little.
        t = u * sx
        pr = np.clip((t + 0.35) / 1.35, 0, 1) ** 2 * np.exp(-(np.maximum(t - 1.0, 0) / 0.8) ** 2)
        vf = np.where(h > 0, np.exp(-(np.maximum(h - 0.35, 0) / 0.8) ** 2), np.exp(-(np.maximum(-h - 0.30, 0) / SM_LO_FALL) ** 2))
        sm = np.clip(rig.smooth(pr * vf * front, 2), 0, 1) * skf
        lat = (1 - smoothstep(0.9, 1.35, np.abs(u))) * front * sideW[s] * skf
        lips = np.exp(-(h / 0.5) ** 2) * lat
        wide = 0.75 + 0.25 * np.clip(t, 0, 1)                    # thinning: centre shares both sides, outer thirds most
        flat = np.zeros((nV, 3))
        g_lo = np.where(h < 0, -h * np.exp(-(h / SM_LO_W) ** 2), 0.0)       # 0 at the seam, peak at the lower border, 0 at chin
        g_up = np.where(h > 0, h * np.exp(-(h / SM_UP_W) ** 2), 0.0)
        thin = rig.smooth(np.c_[g_lo * wide * lat, g_up * wide * lat], 2)
        flat[:, 1] = (SM_THIN_LO * thin[:, 0] - SM_THIN_UP * thin[:, 1]) * mw
        flat[:, 2] = -0.045 * mw * lips
        nl = rig.nasolabial(s)
        fold = nl * gauss(VA, 0.45 * (rig.alar[s] + rig.corner[s]) + 0.1 * rig.cheek[s], 0.65 * mw) * sideW[s] * skf
        fold *= 1 - smoothstep(0.3, 0.9, sm)                     # the corner itself is driven by sm
        pocket = gauss(VA, rig.corner[s] + V3(sx * 0.16, 0.06, 0) * mw, 0.14 * mw) * sideW[s] * skf
        lidD, lof = lower_lid(s, SM_LID)
        apple = gauss(VA, rig.cheek[s] + V3(-sx * 0.05, -0.10, 0) * iod, 0.30 * iod) * cheek_gate[s] * sideW[s] * skf
        D = sm[:, None] * (V3(sx * SM_OUT, SM_UP, -SM_BACK) * mw) + flat + lidD
        D += apple[:, None] * V3(sx * 0.012, SM_APPLE_UP, SM_APPLE_FWD) * iod
        D += fold[:, None] * V3(sx * 0.02, 0.08, SM_FOLD) * mw
        D += pocket[:, None] * V3(0, 0, -SM_POCKET) * mw
        put(f"mouthSmile{side}", D)
        LID[f"mouthSmile{side}"] = (None, lof, 0, lidD)
        # smile on an open jaw: the seam opens faster than the sticky jaw alone (a slightly open smile parts the lips
        # and shows the upper teeth) and the upper lip centre lifts off the incisors
        ulift = rig.upper * np.exp(-(np.maximum(h, 0) / 0.75) ** 2) * np.exp(-(u / 1.1) ** 2) * front * skf
        ulift = np.clip(rig.smooth(ulift, 2), 0, 1) * skf * sideW[s]
        # mouth4: the lower lip centre drops off the upper teeth too (the reference's D-shaped open smile)
        ldrop = rig.lower * np.exp(-(np.minimum(h, 0) / 0.60) ** 2) * np.exp(-(u / 0.95) ** 2) * front * skf
        ldrop = np.clip(rig.smooth(ldrop, 2), 0, 1) * skf
        corr("jawOpen", f"mouthSmile{side}", (SMILE_SEAL * seal + ulift[:, None] * V3(0, SM_TEETH_LIFT, 0) * mw
                                              + ldrop[:, None] * V3(0, -SM_LO_DROP, -0.1 * SM_LO_DROP) * mw)
             * sideW[s][:, None])
        # frown
        D = cf[:, None] * (V3(sx * 0.06, -0.32, -0.02) * mw)
        D += (gauss(VA, rig.corner[s] + V3(sx * 0.1 * mw, -0.7 * mw, 0), 0.55 * mw) * sideW[s] * skf)[:, None] * V3(0, -0.06, 0.02) * mw
        put(f"mouthFrown{side}", D)
        # dimple: corners back + small dent beside the corner
        dent = gauss(VA, rig.corner[s] + V3(sx * 0.35 * mw, 0, 0), 0.16 * mw) * skf
        D = cf[:, None] * (V3(sx * 0.08, 0.02, -0.14) * mw) + dent[:, None] * V3(0, 0, -0.07) * mw
        put(f"mouthDimple{side}", D)
        # stretch: corners out + down, lips thin
        lowlat = (rig.lower * smoothstep(0.1, 0.8, u * sx) * lipband)
        D = cf[:, None] * (V3(sx * 0.24, -0.10, -0.05) * mw) + lowlat[:, None] * V3(0, -0.07, 0) * mw
        put(f"mouthStretch{side}", D)
        # upper lip up (snarl side) / lower lip down
        lat = np.exp(-((u * sx - 0.45) / 0.55) ** 2) * sideW[s]
        upf = rig.smooth((rig.upper * np.exp(-(np.maximum(h, 0) / 0.9) ** 2) * lat * front).astype(float), 1) * skf
        put(f"mouthUpperUp{side}", upf[:, None] * V3(sx * 0.02, UPPER_UP, 0.05) * mw)
        lof2 = rig.smooth((rig.lower * np.exp(-(np.minimum(h, 0) / 0.9) ** 2) * lat * front).astype(float), 1) * skf
        put(f"mouthLowerDown{side}", lof2[:, None] * V3(sx * 0.02, -0.24, 0.04) * mw)
        # nose sneer
        al = rig.alar[s]
        g = (gauss(VA, al + V3(-sx * 0.1 * mw, 0.25 * mw, -0.05 * mw), 0.42 * mw) * sideW[s] * skf)
        lipup = upf * 0.35
        put(f"noseSneer{side}", g[:, None] * V3(-sx * 0.03, 0.17, -0.02) * mw + lipup[:, None] * V3(0, 0.15, 0) * mw)

        # ---- lids
        bl_close = 0.18     # lower lid rises this fraction of the aperture in a blink
        lo_bl, lo_bl_f = lower_lid(s, bl_close)
        ap = rig.aperture(s)
        over = np.radians(1.0)
        f_bl = lambda az, ap=ap: (1 - bl_close) * ap(az) + over
        A, B = rig.lid_delta(s, "up", f_bl, path=True)
        S[f"eyeBlink{side}"] = (A + lo_bl) * skf[:, None]
        corr(f"eyeBlink{side}", f"eyeBlink{side}", B)
        # wide
        upD, upf_w = upper_lid(s, deg=-13.0)
        loD, lof_w = lower_lid(s, -0.05)
        put(f"eyeWide{side}", upD + loD)
        LID[f"eyeWide{side}"] = (upf_w, lof_w, upD, loD)
        # squint: lower lid up a lot, upper down a little, orbital cheek up
        upD, upf_s = upper_lid(s, frac=0.08)
        loD, lof_s = lower_lid(s, 0.25)
        put(f"eyeSquint{side}", upD + loD + cheek_lift(s, 0.35))
        LID[f"eyeSquint{side}"] = (upf_s, lof_s, upD, loD)
        # cheek squint: cheek lift + lower lid push
        loD, lof_c = lower_lid(s, 0.10)
        put(f"cheekSquint{side}", loD + cheek_lift(s, 1.0))
        LID[f"cheekSquint{side}"] = (None, lof_c, 0, loD)
        # happy: arched lower lid + upper lid lowered
        upD, upf_h = upper_lid(s, frac=0.18)
        loD, lof_h = lower_lid(s, 0.30, arch=0.7)
        put(f"eyeHappy{side}", upD + loD + cheek_lift(s, 0.6))
        LID[f"eyeHappy{side}"] = (upf_h, lof_h, upD, loD)
        # look up / down (lid follow ~40% of a ~25 deg gaze), in / out (small yaw of the lid shells)
        upD, upf_u = upper_lid(s, deg=-11.0)
        loD = rig.lid_delta(s, "lo", lambda az: np.full_like(az, np.radians(-5.0)))
        put(f"eyeLookUp{side}", upD + loD)
        LID[f"eyeLookUp{side}"] = (upf_u, lambda az: np.full_like(az, np.radians(-5.0)), upD, loD)
        upD, upf_d = upper_lid(s, deg=13.0)
        loD = rig.lid_delta(s, "lo", lambda az: np.full_like(az, np.radians(5.0)))
        put(f"eyeLookDown{side}", upD + loD)
        LID[f"eyeLookDown{side}"] = (upf_d, lambda az: np.full_like(az, np.radians(5.0)), upD, loD)
        E = rig.eyes[s]
        shell = np.maximum(E["a_up"], E["a_lo"]) * E["taper"]
        for nm, yaw in (("In", -1), ("Out", 1)):
            ang = np.radians(4.0) * yaw * sx * shell
            D = rotate(VA, E["c"], V3(0, 1, 0), ang) - VA
            put(f"eyeLook{nm}{side}", D)
        # blink correctives: close onto the other shape's lower lid, cancel its own upper-lid move
        # exact at (blink=1, X=1): the composed rotations minus the sum of the separate linear deltas
        bl_full = S[f"eyeBlink{side}"] + B
        for X, (fu, fl, xu, xl) in LID.items():
            if not X.endswith(side):
                continue
            t_up = lambda az, fl=fl: f_bl(az) + fl(az)
            t_lo = lambda az, fl=fl: lo_bl_f(az) + fl(az)
            tgt = rig.lid_delta(s, "up", t_up) + rig.lid_delta(s, "lo", t_lo)
            corr(f"eyeBlink{side}", X, tgt - bl_full - xu - xl)

        # ---- brows (skin under the strand brows): MH brow units as the topology mask
        bm = np.linalg.norm(rig.ut(f"eyebrows-{'left' if s == 'l' else 'right'}-down"), axis=1)
        bm = np.clip(bm / max(bm[skin].max(), 1e-9) * 1.6, 0, 1)
        bm = np.clip(rig.smooth(bm, 3), 0, 1) * sideW[s] * skf
        t = sx * (x - E["c"][0]) / R
        inner = 1 - smoothstep(-0.7, 0.5, t)
        outer = smoothstep(-0.3, 0.9, t)
        Ra = 0.19 * iod                                      # brow amplitude scale (face, not the stylised eyeball)
        brow_mid = gauss(VA, E["c"] + V3(0, 1.45 * R, 0.2 * R), 1.6 * R) * sideW[s] * skf
        bmask = np.maximum(bm, 0.7 * brow_mid * (E["r"] > 1.25 * R))
        S_down = bmask[:, None] * V3(-sx * 0.14, -0.40, 0.07) * Ra * (0.6 + 0.4 * inner)[:, None]
        put(f"browDown{side}", S_down)
        put(f"browOuterUp{side}", (bmask * outer)[:, None] * V3(sx * 0.02, 0.30, -0.02) * Ra)
        put(f"browTilt{side}", (bmask * inner)[:, None] * V3(-sx * 0.03, 0.30, 0.0) * Ra
            + (bmask * outer)[:, None] * V3(0, -0.16, 0.0) * Ra)
        if s == "l":
            inner_up = np.zeros((nV, 3))
        inner_up += (bmask * inner)[:, None] * V3(sx * 0.02, 0.34, 0.0) * Ra
    put("browInnerUp", inner_up)

    # smile2: combination guard for the open smile. Sticky lips lag the jaw at small jawOpen while the lower gums ride the
    # jaw bone; with the smile's thinned/pulled-back lower lip the gums came through the skin below the lip
    # (smile 1 + jawOpen .15). Fix the pose (both smiles 1, jawOpen JG) against the teeth at that jaw angle and put the
    # forward push, / JG, into corr_jawOpen_mouthSmileL/R (weight = jawOpen * smile).
    for JG in SMILE_GUARD_J:
        P0 = VA + JG * S["jawOpen"] + JG * JG * S["corr_jawOpen_jawOpen"]
        Dsm = S["mouthSmileLeft"] + S["mouthSmileRight"] + JG * (S["corr_jawOpen_mouthSmileLeft"] + S["corr_jawOpen_mouthSmileRight"])
        Dg = guard(Dsm, P0=P0, zf=front_map(JG)) - Dsm
        for side, s_, sg in SIDES:
            S[f"corr_jawOpen_mouthSmile{side}"] = S[f"corr_jawOpen_mouthSmile{side}"] + Dg * (sideW[s_] * skf)[:, None] / JG

    # ------------------------------------------------------------------ central mouth shapes
    contract = np.exp(-(u / 1.5) ** 2) * np.exp(-(h / 0.9) ** 2) * front * skf
    lipfw = lipband * (1 - smoothstep(1.0, 1.4, np.abs(u)))
    D = np.zeros((nV, 3))
    D[:, 0] = -0.55 * x * contract
    D[:, 2] = 0.50 * mw * lipfw + 0.12 * mw * mouthR
    put("mouthPucker", D)
    corr("jawOpen", "mouthPucker", -0.35 * J * (lipfw * np.exp(-(u / 0.8) ** 2))[:, None])
    D = np.zeros((nV, 3))
    D[:, 0] = -FUNNEL_NARROW * x * contract
    opn = lipfw * np.exp(-(u / 0.75) ** 2)
    D[:, 1] = np.where(rig.upper, FUNNEL_OPEN_UP, -FUNNEL_OPEN_LO) * mw * opn
    D[:, 2] = 0.36 * mw * lipfw
    put("mouthFunnel", D)
    sh = np.exp(-((VA - mo) ** 2).sum(1) / (2 * (1.3 * mw) ** 2)) * front * skf
    put("mouthLeft", sh[:, None] * V3(0.32, 0.02, -0.03) * mw)
    put("mouthRight", sh[:, None] * V3(-0.32, 0.02, -0.03) * mw)
    put("mouthShrugLower", (np.exp(-((h + 0.55) / 0.75) ** 2) * front * skf)[:, None] * V3(0, 0.13, 0.0) * mw
        + (np.exp(-((h + 0.25) / 0.45) ** 2) * front * skf * (1 - smoothstep(0.8, 1.3, np.abs(u))))[:, None] * V3(0, 0, 0.12) * mw)
    put("mouthShrugUpper", (np.exp(-((h - 0.25) / 0.55) ** 2) * front * skf * (1 - smoothstep(0.9, 1.4, np.abs(u))))[:, None]
        * V3(0, 0.10, 0.07) * mw)

    put("mouthRollUpper", roll("upper", 38))
    put("mouthRollLower", roll("lower", 42))
    put("lipTuckUpper", roll("upper", 55))
    put("lipTuckLower", roll("lower", 62))

    # labiodental: lower lip up/back under the upper incisors, upper lip lifts a little
    ut_v = base.verts("helper-upper-teeth")
    P = VA[ut_v]
    mid = P[np.abs(P[:, 0]) < 0.35 * mw]
    t_tip = np.array([0, mid[:, 1].min(), mid[:, 2].max()])
    sl = rig.seam_l[np.argsort(np.abs(VA[rig.seam_l, 0]))[:3]]
    lpt = VA[sl].mean(0)
    tgt = np.array([0, t_tip[1] + 0.0004, t_tip[2] + 0.0012])
    dv = tgt - lpt
    dv[1] = min(dv[1], 0.0)
    lw = rig.lower * np.exp(-(np.minimum(h, 0) / 0.6) ** 2) * np.exp(-(u / 1.15) ** 2) * front
    lw = np.clip(rig.smooth(lw.astype(float), 2), 0, 1) * skf
    upw = rig.upper * np.exp(-(np.maximum(h, 0) / 0.7) ** 2) * np.exp(-(u / 1.0) ** 2) * front * skf
    put("labioDental", lw[:, None] * dv[None] + upw[:, None] * V3(0, FV_UP_LIFT, 0.03) * mw)   # smile2: upper lip lifts so the incisors rest visibly on the lower lip

    # cheek puff (both sides) with lips sealed and pushed forward a touch
    puff = np.zeros((nV, 3))
    for side, s, sg in SIDES:
        pc = rig.corner[s] + V3(sg * 0.55 * mw, 0.12 * mw, -0.40 * mw)
        g = gauss(VA, pc, 0.62 * mw) * sideW[s] * skf * (1 - c_seal)
        puff += g[:, None] * base.N * 0.30 * mw
    puff += (lipband * np.exp(-(u / 1.0) ** 2))[:, None] * V3(0, 0, 0.06) * mw
    put("cheekPuff", puff)

    for n in ("tongueOut", "tongueTipUp", "tongueDental", "tongueBack"):
        S[n] = np.zeros((nV, 3))
    build.rig = rig
    build.c_seal = c_seal
    return S


def jaw_open_deg(base, S=None):
    """Jaw-bone angle (deg, about the jaw's local +x) that matches jawOpen exactly (it is that rotation)."""
    rig = getattr(build, "rig", None) or FaceRig(base)
    return float(JAW_OPEN_DEG * _jaw_sign(rig))


def weights_with_correctives(w):
    """w: {shape: weight} -> adds corr_* weights (product rule) for the correctives defined by the last build()."""
    out = dict(w)
    for nm, (a, b) in CORRECTIVES.items():
        out[nm] = w.get(a, 0.0) * w.get(b, 0.0)
    return out
