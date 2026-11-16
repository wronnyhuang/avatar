"""HEAD-MTEX — junor eyeball maps (iris + sclera), procedural. Same layout / outputs as juno's eyes_tex.eye_maps
(albedo RGBA, normal RGB, packed RGBA, 0..255; uv centre = the iris, u = horizontal), so the asset params
(iris_uv_radius, pupil_uv_radius, limbus_width) and SHADE's mat_eye_pbr are unchanged.

The iris is built in POLAR space (angle x radius) and resampled: radial stromal fibres are noise stretched along the
radius with a slow angular wander (they curve like real trabeculae), several octaves; crypts of Fuchs (dark lens-shaped
lacunae, densest just outside the collarette); the collarette (a wavy raised lighter ring ~1/3 out); the pupillary
ruff (a thin dark fringe at the pupil); faint concentric contraction furrows in the outer third; a darker limbal ring.
Colour: lighter amber round the pupil -> amber-brown stroma -> warm brown outer -> dark limbal ring.
SHADE's iris grade (mesh.pbr.eye.iris_grade, 1 for junor) keeps the map's LUMINANCE structure, compressed
(lum/0.16)^0.45 clamped .55..1.35, on its own amber-brown ramp -> the map's luminance is kept in ~0.05..0.30 linear
in the stroma so the structure is not clipped; the hue matters when the grade < 1.
Sclera: slightly warm off-white, a faint yellow-pink tint and fine branching vessels toward the two corners
(horizontal), almost none near the cornea/top/bottom (SHADE adds its own faint shader veins on top).
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage


def _sm(a, b, x):
    t = np.clip((x - a) / (b - a), 0, 1)
    return t * t * (3 - 2 * t)


def _pnoise(rng, NA, NR, sa, sr):
    """polar noise, wrapping in angle: sa / sr = blur sigma (px) along angle / radius."""
    a = rng.standard_normal((NR, NA))
    a = ndimage.gaussian_filter(a, (sr, sa), mode=("nearest", "wrap"))
    return a / max(a.std(), 1e-9)


def eye_maps(n, iris_uv_r, pupil_uv_r, seed=11):
    rng = np.random.default_rng(seed)
    NA, NR = 2048, 320
    tp = pupil_uv_r / iris_uv_r
    TR = 1.12                                            # polar grid covers t in [0, TR]
    tt = (np.arange(NR) + 0.5) / NR * TR                 # (NR,)
    T = tt[:, None] * np.ones((1, NA))
    A = (np.arange(NA) + 0.5) / NA * 2 * np.pi - np.pi
    A = np.ones((NR, 1)) * A[None]
    # ---------- radial position remapped to the stroma (0 pupil edge .. 1 limbus)
    s = np.clip((T - tp) / (1 - tp), 0, 1.2)
    # angular wander: fibres curve slowly
    wander = _pnoise(rng, NA, NR, 60, 40) * 6.0
    def warped(sa, sr, amp=1.0):
        f = _pnoise(rng, NA, NR, sa, sr)
        # shift each row in angle by the wander (px) -> curved streaks
        out = np.empty_like(f)
        for i in range(NR):
            out[i] = np.interp(np.arange(NA) + wander[i] * amp, np.arange(NA + 1), np.r_[f[i], f[i, :1]], period=NA)
        return out
    fib1 = warped(0.9, 14)           # fine trabeculae
    fib2 = warped(2.2, 24)
    fib3 = warped(6.0, 40, 0.6)      # coarse bundles
    fib = 0.45 * fib1 + 0.35 * fib2 + 0.35 * fib3
    ridge = 1 - np.abs(np.tanh(0.9 * fib1))                     # bright thin ridges where fine noise crosses zero
    # collarette: wavy ring ~ 0.32 of the stroma out (the boundary pupillary / ciliary zone)
    cpos = 0.30 + 0.05 * _pnoise(rng, NA, 1, 40, 0)[0][None] + 0.02 * _pnoise(rng, NA, 1, 10, 0)[0][None]
    coll = np.exp(-((s - cpos) / 0.045) ** 2)
    inner = _sm(cpos + 0.03, cpos - 0.04, s)                    # pupillary zone (inside the collarette)
    # crypts: lens-shaped dark pits, densest just outside the collarette, a few in the periphery
    crypt = np.zeros((NR, NA))
    for _ in range(110):
        a0 = rng.uniform(0, NA)
        s0 = cpos[0, int(a0) % NA] + abs(rng.normal(0.06, 0.12))
        if s0 > 0.95:
            continue
        da = (np.arange(NA) - a0 + NA / 2) % NA - NA / 2
        wa = rng.uniform(5, 20) * (0.6 + s0)
        wr = rng.uniform(0.025, 0.07)
        crypt += np.exp(-(da[None] / wa) ** 2 - ((s - s0) / wr) ** 2) * rng.uniform(0.5, 1.0)
    crypt = np.clip(crypt, 0, 1)
    # contraction furrows: faint concentric dark arcs in the outer third (broken, not full rings)
    fur = np.zeros((NR, NA))
    for f0 in (0.66, 0.76, 0.86):
        brk = _sm(-0.2, 0.6, _pnoise(rng, NA, 1, 50, 0)[0])[None]
        fur += np.exp(-((s - f0 - 0.012 * wander / 6) / 0.010) ** 2) * brk
    ruff = np.exp(-((s - 0.006) / 0.012) ** 2)                  # pupillary ruff (pigment epithelium fringe)
    limb = _sm(0.80, 0.985, s)
    # ---------- colour (sRGB 0..1)
    gold = np.array([0.80, 0.56, 0.25])          # lighter amber round the pupil
    amber = np.array([0.52, 0.32, 0.14])         # amber-brown stroma
    outer = np.array([0.40, 0.23, 0.11])         # warm brown outer
    rimc = np.array([0.20, 0.12, 0.07])          # limbal ring
    w1 = _sm(cpos + 0.0, cpos + 0.22, s)
    col = gold * (1 - w1)[..., None] + amber * w1[..., None]
    w2 = _sm(0.45, 0.85, s)
    col = col * (1 - w2)[..., None] + outer * w2[..., None]
    # pupillary zone flecks of gold over the collarette
    col = col * (1 + 0.22 * fib[..., None] * np.array([1.0, 0.95, 0.8]) * (0.6 + 0.4 * inner[..., None]))
    col = col + 0.10 * ridge[..., None] * np.array([1.0, 0.75, 0.4]) * (0.3 + 0.7 * _sm(0.1, 0.5, s))[..., None]
    col = col + 0.09 * coll[..., None] * np.array([1.0, 0.8, 0.45])
    col = col * (1 - 0.45 * crypt)[..., None]
    col = col * (1 - 0.25 * np.clip(fur, 0, 1))[..., None]
    col = col * (1 - 0.6 * ruff)[..., None] + np.array([0.10, 0.06, 0.04]) * 0.6 * ruff[..., None]
    col = col * (1 - 0.7 * limb)[..., None] + rimc * 0.7 * limb[..., None]
    hI = (0.5 * fib3 + 0.35 * fib1 + 1.6 * coll - 1.4 * crypt - 0.5 * np.clip(fur, 0, 1)) * (T < 1.0)
    # ---------- polar -> cartesian (n x n eyeball uv)
    yy, xx = np.mgrid[0:n, 0:n]
    u, v = (xx + 0.5) / n - 0.5, (yy + 0.5) / n - 0.5
    rr, ang = np.sqrt(u * u + v * v), np.arctan2(v, u)
    t = rr / iris_uv_r
    ri = np.clip(t / TR * NR - 0.5, 0, NR - 1)
    ai = (ang + np.pi) / (2 * np.pi) * NA - 0.5
    inside = t < TR
    icol = np.zeros((n, n, 3))
    for c in range(3):
        icol[..., c] = ndimage.map_coordinates(col[..., c], [ri, ai], order=1, mode="grid-wrap")
    hh = ndimage.map_coordinates(hI, [ri, ai], order=1, mode="grid-wrap") * inside
    # ---------- sclera
    sc = np.array([0.93, 0.905, 0.865])
    back = _sm(iris_uv_r * 1.15, 0.5, rr)
    corner = np.abs(np.cos(ang)) ** 3                   # toward the inner / outer canthus (horizontal)
    scl = sc * (1 - 0.08 * back)[..., None]
    scl = scl * (1 - 0.10 * (back * corner)[..., None] * np.array([0.0, 0.35, 0.8]))   # faint warm/yellow-pink at the corners
    veins = np.zeros((n, n))
    for _ in range(46):
        a = rng.choice([0.0, np.pi]) + rng.normal(0, 0.55)        # start mostly at the corners
        r0 = 0.49
        x, y = 0.5 + r0 * np.cos(a), 0.5 + r0 * np.sin(a)
        w = rng.uniform(0.4, 1.0)
        steps = int(rng.uniform(0.15, 0.42) * n)
        ad = a
        for _s in range(steps):
            ad += rng.normal(0, 0.09)
            d = np.array([0.5 - x, 0.5 - y]); d /= np.linalg.norm(d) + 1e-9
            x += (0.55 * d[0] + 0.45 * np.cos(ad + np.pi)) / n
            y += (0.55 * d[1] + 0.45 * np.sin(ad + np.pi)) / n
            if np.hypot(x - 0.5, y - 0.5) < iris_uv_r * (1.35 + 0.4 * rng.random()):
                break
            veins[int(np.clip(y * n, 0, n - 1)), int(np.clip(x * n, 0, n - 1))] += w
            if rng.random() < 0.004:      # a branch: restart nearby at a new heading
                ad += rng.choice([-1, 1]) * 0.8
    veins = ndimage.gaussian_filter(veins, n / 1024 * 0.8) * 3.0
    veins = np.clip(veins, 0, 1) * (0.25 + 0.75 * back) * (0.3 + 0.7 * corner)
    scl = scl * (1 - 0.38 * veins)[..., None] + veins[..., None] * np.array([0.72, 0.30, 0.28]) * 0.38
    # iris -> sclera across the limbus (soft), a faint grey ring just outside
    m = _sm(1.0, 1.07, t)
    col2 = icol * (1 - m)[..., None] + scl * m[..., None]
    col2 = col2 * (1 - 0.10 * np.exp(-((t - 1.08) / 0.06) ** 2))[..., None]
    pm = _sm(tp * 0.97, tp * 1.03, t)
    col2 = col2 * pm[..., None] + np.array([0.012, 0.010, 0.010]) * (1 - pm)[..., None]
    alb = np.concatenate([np.clip(col2, 0, 1) * 255, np.full((n, n, 1), 255.0)], -1)
    # ---------- normal (iris relief; sclera flat)
    h = ndimage.gaussian_filter(hh * _sm(tp, tp + 0.05, t), n / 1024 * 0.6)
    gy, gx = np.gradient(h)
    k = 0.5
    nx, ny = -gx * k * n / 256, -gy * k * n / 256
    nz = np.ones_like(nx)
    ln = np.sqrt(nx * nx + ny * ny + nz * nz)
    nrm = np.stack([nx / ln, ny / ln, nz / ln], -1) * 127.5 + 127.5
    cry_c = ndimage.map_coordinates(crypt, [ri, ai], order=1, mode="grid-wrap") * inside
    lim_c = _sm(0.82, 0.98, (t - tp) / (1 - tp))
    ao = 1 - 0.3 * cry_c * (t < 1) - 0.15 * lim_c * (t < 1)
    rough = np.where(t < 1.0, 0.45, 0.30)
    spec = np.where(t < 1.0, 0.08, 0.30) * _sm(tp * 0.9, tp * 1.1, t)
    thick = np.where(t < 1.0, 0.2, 0.55)
    pk = np.stack([ao, rough, spec, thick], -1) * 255
    return alb, nrm, pk
