"""asset — procedural eye textures (painted animated-film amber iris + sclera), occlusion and tearline strips.

UV layout of the eyeball: polar, centre (0.5, 0.5) = front pole, uv radius = 0.5 * theta / pi (see eyes_geo.eye_uv).
"""
from __future__ import annotations

import numpy as np

from juno_base import smoothstep


def _fourier_angle(ang, t, rng, kmin, kmax, n, drift=0.0):
    """sum of random angular harmonics (radial streaks), phases drifting slowly with radius t."""
    out = np.zeros_like(ang)
    for _ in range(n):
        k = rng.integers(kmin, kmax)
        out += np.sin(k * ang + rng.uniform(0, 2 * np.pi) + drift * rng.normal() * t) / np.sqrt(k)
    return out / np.sqrt(n) * 2


def _blur(a, r):
    if r <= 0:
        return a
    from scipy.ndimage import gaussian_filter
    return gaussian_filter(a, r)


def eye_maps(n, iris_uv_r, pupil_uv_r, seed=7):
    """-> albedo RGBA (n,n,4) 0..255, normal RGB (n,n,3), packed RGBA (n,n,4)."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:n, 0:n]
    u, v = (xx + 0.5) / n - 0.5, (yy + 0.5) / n - 0.5
    rr, ang = np.sqrt(u * u + v * v), np.arctan2(v, u)
    t = rr / iris_uv_r                                   # 0 centre .. 1 limbus
    tp = pupil_uv_r / iris_uv_r
    # ---- iris colour layers (linear-ish sRGB 0..1)
    # asset-eyes2: amber-BROWN (concept), less orange: warm gold ring round the pupil, brown body, dark rim
    deep = np.array([0.27, 0.13, 0.05])      # outer brown
    amber = np.array([0.60, 0.34, 0.12])     # mid amber-brown
    gold = np.array([0.78, 0.53, 0.22])      # collarette / ring round the pupil
    s1 = smoothstep(tp, tp + 0.22, t)        # pupil edge -> collarette
    s2 = smoothstep(0.50, 0.95, t)           # collarette -> outer
    col = gold * (1 - s1)[..., None] + amber * s1[..., None]
    col = col * (1 - s2)[..., None] + deep * s2[..., None]
    # radial fibres: fine + coarse streaks, brighter ridges
    fine = _fourier_angle(ang, t, rng, 60, 260, 60, drift=2.0)
    coarse = _fourier_angle(ang, t, rng, 12, 40, 14, drift=1.0)
    fib = 0.55 * fine + 0.45 * coarse
    col = col * (1 + 0.16 * fib)[..., None]
    # collarette ridge: wavy ring at t ~ 0.42
    cr = 0.42 + 0.035 * _fourier_angle(ang, 0, rng, 5, 14, 6)
    ridge = np.exp(-((t - cr) / 0.045) ** 2)
    col = col + ridge[..., None] * np.array([0.10, 0.08, 0.03])
    # crypts: dark radial-elongated spots between collarette and limbus
    crypt = np.zeros((n, n))
    for _ in range(70):
        a0, t0 = rng.uniform(-np.pi, np.pi), rng.uniform(0.5, 0.85)
        da = np.angle(np.exp(1j * (ang - a0)))
        crypt += np.exp(-(da / rng.uniform(0.02, 0.05)) ** 2 - ((t - t0) / rng.uniform(0.04, 0.09)) ** 2)
    crypt = np.clip(crypt, 0, 1)
    col = col * (1 - 0.35 * crypt)[..., None]
    # sparse lighter flecks
    fl = (rng.random((n, n)) > 0.99965).astype(float)
    fl = _blur(fl, n / 1024 * 1.2) * 40
    col = col + np.clip(fl, 0, 1)[..., None] * np.array([0.12, 0.09, 0.04]) * ((t > 0.3) & (t < 0.85))[..., None]
    # dark limbal ring (soft outer edge)
    limb = smoothstep(0.78, 0.96, t)
    col = col * (1 - 0.78 * limb)[..., None]
    # ---- sclera: warm off-white, faint warm edge toward the back, faintest veins
    sc = np.array([0.93, 0.90, 0.86])
    back = smoothstep(iris_uv_r * 1.2, 0.5, rr)
    scl = sc * (1 - 0.10 * back)[..., None] + np.array([0.04, -0.01, -0.03]) * back[..., None]
    veins = np.zeros((n, n))
    for _ in range(16):                       # random walks from the back toward the limbus
        a, r0 = rng.uniform(-np.pi, np.pi), 0.49
        x, y = 0.5 + r0 * np.cos(a), 0.5 + r0 * np.sin(a)
        steps = int(rng.uniform(0.25, 0.6) * n)
        for _s in range(steps):
            a += rng.normal(0, 0.06)
            d = np.array([0.5 - x, 0.5 - y])
            d /= np.linalg.norm(d) + 1e-9
            x += (0.6 * d[0] + 0.4 * np.cos(a)) / n
            y += (0.6 * d[1] + 0.4 * np.sin(a)) / n
            if np.hypot(x - 0.5, y - 0.5) < iris_uv_r * 1.25:
                break
            veins[int(np.clip(y * n, 0, n - 1)), int(np.clip(x * n, 0, n - 1))] += 1
    veins = np.clip(_blur(veins, n / 1024 * 1.4) * 2, 0, 1) * (0.15 + 0.85 * back)
    scl = scl * (1 - 0.30 * veins)[..., None] + veins[..., None] * np.array([0.78, 0.36, 0.34]) * 0.30
    # limbus blend iris -> sclera (soft, slightly grey-blue ring outside)
    m = smoothstep(1.0, 1.06, t)
    col = col * (1 - m)[..., None] + scl * m[..., None]
    shadow = np.exp(-((t - 1.08) / 0.07) ** 2) * 0.10
    col = col * (1 - shadow)[..., None]
    # pupil (soft 1-2 px edge)
    pm = smoothstep(tp * 0.97, tp * 1.03, t)
    col = col * pm[..., None] + np.array([0.012, 0.010, 0.010]) * (1 - pm)[..., None]
    alb = np.concatenate([np.clip(col, 0, 1) * 255, np.full((n, n, 1), 255.0)], -1)
    # ---- normal: iris relief (fibres + collarette + crypt dips), flat sclera
    h = (0.5 * fib + 1.5 * ridge - 1.2 * crypt) * (t < 1.0) * smoothstep(tp, tp + 0.05, t)
    h = _blur(h, n / 1024 * 0.7)
    # gradient in uv space; scale chosen for subtle relief
    gy, gx = np.gradient(h)
    k = 0.6
    nx, ny = -gx * k * n / 256, -gy * k * n / 256
    nz = np.ones_like(nx)
    ln = np.sqrt(nx * nx + ny * ny + nz * nz)
    nrm = np.stack([nx / ln, ny / ln, nz / ln], -1) * 127.5 + 127.5
    # ---- packed: AO (crypts / limbus slightly), roughness, spec, thickness
    ao = 1 - 0.25 * crypt * (t < 1) - 0.15 * limb * (t < 1)
    rough = np.where(t < 1.0, 0.45, 0.30)
    spec = np.where(t < 1.0, 0.08, 0.30) * smoothstep(tp * 0.9, tp * 1.1, t)   # the cornea coat carries the gloss
    thick = np.where(t < 1.0, 0.2, 0.55)
    pk = np.stack([ao, rough, spec, thick], -1) * 255
    return alb, nrm, pk


def strip_occlusion(nu=512, nv=64, phi_top=np.pi / 2):
    """eye occlusion strip: u = phi/2pi along the lid line, v = 0 at the lid .. 1 inner edge. Dark warm AO, darker under
    the upper lid (reads as the lash line's shadow + the lid thickness)."""
    u = (np.arange(nu) + 0.5) / nu
    v = (np.arange(nv) + 0.5) / nv
    U, V = np.meshgrid(u, v, indexing="xy")
    phi = U * 2 * np.pi
    top = np.clip(np.sin(phi), 0, 1)            # 1 at the top of the eye
    bot = np.clip(-np.sin(phi), 0, 1)
    reach = 0.40 + 0.60 * top                    # AO reaches further onto the eye under the upper lid
    a = (1 - smoothstep(0.0, reach, V)) ** 1.3
    a = a * (0.55 + 0.42 * top ** 0.7 - 0.3 * bot)
    a = a * (1 - 0.5 * np.abs(np.cos(phi)) ** 6)   # asset-eyes3: lighter at the canthi (dirty corner patches)
    a = a * smoothstep(0.0, 0.05, V)             # fade the very outer row (under the lid) to avoid a seam
    a = np.clip(a + 0.0, 0, 1)
    col = np.zeros((nv, nu, 4))
    col[..., :3] = np.array([0.10, 0.05, 0.04]) * 255
    col[..., 3] = a * 255
    return col


def strip_tearline(nu=512, nv=32):
    """tearline: u along the lower lid (0 lateral .. 1 past the medial corner), v across. Wet, mostly clear; a bit
    pinker toward the caruncle (u > 0.85)."""
    u = (np.arange(nu) + 0.5) / nu
    v = (np.arange(nv) + 0.5) / nv
    U, V = np.meshgrid(u, v, indexing="xy")
    across = np.exp(-((V - 0.45) / 0.28) ** 2)
    ends = smoothstep(0.0, 0.08, U)
    car = smoothstep(0.82, 0.95, U)
    a = across * ends * (0.35 + 0.45 * car)
    col = np.zeros((nv, nu, 4))
    col[..., :3] = (np.array([0.95, 0.85, 0.82]) * (1 - car)[..., None] + np.array([0.85, 0.45, 0.45]) * car[..., None]) * 255
    col[..., 3] = a * 255
    return col
