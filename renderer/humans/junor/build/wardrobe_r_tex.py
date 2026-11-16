"""junor — cloth textures for the realistic wardrobe (copy of juno/build/wardrobe_tex.py; juno's stays untouched).
Round 4 (lead, AAA judge: "repeating herringbone texture stretched and tiled with no fibre depth"): the knit and rib are
BAKED (the shader weave is off for them: weave "none"), on big tiles with a domain-warped stitch grid (rows wander,
stitches vary), yarn ply twist, a fibre halo, per-stitch tone and two low-frequency variations; 1x1 rib for the bands;
cotton poplin for the blouse; teal twill trousers (the shader keeps a fine plain/twill weave + fold AO for those two).

Original juno docstring:
Procedural, seamlessly tiling cloth / leather / button textures for Juno's wardrobe.

Pure numpy (scipy only for a periodic KD-tree in the leather grain). Every map is built
from coordinates that are periodic over one UV tile (integer stitch / thread / cell counts,
wrap-around gradients and FFT blurs), so albedo, normal and packed maps all tile REPEAT.

Map contract
  albedo : sRGB RGBA, A = 255
  normal : linear RGB, tangent space, OpenGL (green up), 128 = 0.
           +x (red) = +u = image right; +y (green) = +v = image UP (PIL row 0 is the top).
  packed : linear RGBA - R ambient occlusion (1 = open), G roughness, B specular level (0.5),
           A sss thickness.

UVs: the caller sets garment UV = surface metres / TILE_M[kind], wrap repeat.
"""
from __future__ import annotations

import os

import numpy as np
from PIL import Image

TILE_M = {"knit": 0.24, "rib": 0.12, "blouse": 0.04, "trousers": 0.06, "leather": 0.05, "buttons": 0.02}

# ----------------------------------------------------------------------------- helpers


def _srgb_to_lin(c):
    c = np.asarray(c, np.float64) / 255.0
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _lin_to_srgb8(c):
    c = np.clip(c, 0.0, 1.0)
    s = np.where(c <= 0.0031308, c * 12.92, 1.055 * np.power(c, 1 / 2.4) - 0.055)
    return np.clip(np.round(s * 255.0), 0, 255).astype(np.uint8)


def _grid(res, tile_mm):
    """Pixel-centre coordinates in mm; x to the right, y DOWN (image rows)."""
    p = (np.arange(res) + 0.5) * (tile_mm / res)
    x, y = np.meshgrid(p, p)
    return x, y


def _fft_freqs(res):
    f = np.fft.fftfreq(res)
    return np.meshgrid(f, f)


def _blur(a, sigma_px):
    """Periodic gaussian blur via FFT."""
    if sigma_px <= 0:
        return a
    fx, fy = _fft_freqs(a.shape[0])
    g = np.exp(-2.0 * (np.pi * sigma_px) ** 2 * (fx ** 2 + fy ** 2))
    return np.real(np.fft.ifft2(np.fft.fft2(a) * g))


def _noise(rng, res, sigma_px, aniso=(1.0, 1.0)):
    """Periodic band-limited noise, zero mean, unit std. aniso stretches (x, y) correlation."""
    w = rng.standard_normal((res, res))
    fx, fy = _fft_freqs(res)
    g = np.exp(-2.0 * (np.pi * sigma_px) ** 2 * ((fx * aniso[0]) ** 2 + (fy * aniso[1]) ** 2))
    n = np.real(np.fft.ifft2(np.fft.fft2(w) * g))
    n -= n.mean()
    return n / (n.std() + 1e-12)


def _normal_from_height(h_mm, px_mm, strength=1.0):
    """OpenGL tangent normal. dh/du from columns; v points up, image rows go down."""
    dhdx = (np.roll(h_mm, -1, 1) - np.roll(h_mm, 1, 1)) / (2 * px_mm)
    dhdy_down = (np.roll(h_mm, -1, 0) - np.roll(h_mm, 1, 0)) / (2 * px_mm)
    dhdv = -dhdy_down
    nx, ny, nz = -dhdx * strength, -dhdv * strength, np.ones_like(h_mm)
    ln = np.sqrt(nx * nx + ny * ny + nz * nz)
    n = np.stack([nx / ln, ny / ln, nz / ln], -1)
    return np.clip(np.round((n * 0.5 + 0.5) * 255), 0, 255).astype(np.uint8)


def _cavity_ao(h_mm, px_mm, radii_mm, depth_mm, floor=0.25):
    """AO from height: how far a point sits below its periodic neighbourhood average."""
    occ = np.zeros_like(h_mm)
    for r in radii_mm:
        occ += np.clip(_blur(h_mm, r / px_mm) - h_mm, 0, None)
    occ /= len(radii_mm)
    return np.clip(1.0 - occ / depth_mm, floor, 1.0)


def _save(texdir, mid, albedo_lin, normal8, ao, rough, spec, thick):
    res = albedo_lin.shape[0]
    a8 = _lin_to_srgb8(albedo_lin)
    alb = np.dstack([a8, np.full((res, res), 255, np.uint8)])
    packed = np.dstack([np.clip(ao, 0, 1), np.clip(rough, 0, 1),
                        np.broadcast_to(spec, ao.shape), np.broadcast_to(thick, ao.shape)])
    packed8 = np.clip(np.round(packed * 255), 0, 255).astype(np.uint8)
    imgs = {"albedo": Image.fromarray(alb, "RGBA"), "normal": Image.fromarray(normal8, "RGB"),
            "packed": Image.fromarray(packed8, "RGBA")}
    os.makedirs(texdir, exist_ok=True)
    for k, im in imgs.items():
        im.save(os.path.join(texdir, f"{mid}_{k}.png"), compress_level=3)
        im1k = im if res <= 1024 else im.resize((1024, 1024), Image.LANCZOS)
        im1k.save(os.path.join(texdir, f"{mid}_{k}_1k.png"), compress_level=3)


# ----------------------------------------------------------------------------- knit


def _knit_legs(lx, ly, W, H, R, h_amp, side):
    """One leg of a stockinette V in cell-local mm (origin = cell centre, y down).
    side = -1 left leg, +1 right leg; the leg's top leans outward so the V points DOWN.
    The leg is a plump tilted ellipse (a yarn loop seen from the front): both ends dive
    behind the neighbouring rows, the bottom (V tip) end sits a little prouder.
    Returns height, along (mm from the bottom end), across (-1..1, + = outward)."""
    th = np.deg2rad(26.0)
    cx, cy = side * 0.245 * W, 0.02 * H
    ux, uy = side * np.sin(th), -np.cos(th)          # axis, pointing up and outward
    a, b = 0.64 * H, R
    qx, qy = lx - cx, ly - cy
    al = qx * ux + qy * uy
    ac = (qx * (-uy) + qy * ux) * side                # + = outward
    # bean shape: centre line bows outward a little
    ac = ac - 0.10 * b * (1 - (al / a) ** 2)
    d2 = (al / a) ** 2 + (ac / b) ** 2
    prof = np.sqrt(np.clip(1.0 - d2, 0.0, None)) ** 0.8
    taper = 0.86 - 0.14 * np.clip(al / a, -1, 1)     # bottom end prouder than the top end
    h = h_amp * prof * taper
    h = np.where(d2 < 1.0, h, -1.0)
    return h, al + a, np.clip(ac / b, -1, 1)


def _stitch_field(x, y, W, H, ncol, nrow, rng, h_amp, R, col_kind=None, purl_drop=0.0):
    """Evaluate the V stitches (and purl bumps for col_kind == 'p') of a knit grid.
    Returns dict with height, ply phase, per-loop id and loop mask."""
    res = x.shape[0]
    c = np.floor(x / W).astype(np.int64)
    r = np.floor(y / H).astype(np.int64)
    lx0 = x - (c + 0.5) * W
    ly0 = y - (r + 0.5) * H
    stitch_val = rng.normal(0, 1, (nrow, ncol))
    leg_val = rng.normal(0, 1, (nrow, ncol, 2))
    leg_phase = rng.uniform(0, 1, (nrow, ncol, 2))
    hbest = np.zeros((res, res))
    ply = np.ones((res, res))
    across = np.zeros((res, res))
    sval = np.zeros((res, res))
    lval = np.zeros((res, res))
    is_purl = np.zeros((res, res), bool)
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            cc = (c + dc) % ncol
            rr = (r + dr) % nrow
            lx = lx0 - dc * W
            ly = ly0 - dr * H
            kind = None if col_kind is None else col_kind[cc]
            # jitter per stitch: hand-knit unevenness
            jx = 0.05 * W * np.tanh(stitch_val[rr, cc])
            jy = 0.04 * H * np.tanh(leg_val[rr, cc, 0])
            for k, side in enumerate((-1, 1)):
                h, a, u = _knit_legs(lx - jx, ly - jy, W, H, R, h_amp * (1 + 0.06 * leg_val[rr, cc, k]), side)
                if kind is not None:
                    h = np.where(kind == 0, h, 0.0)
                # twisted plies: diagonal ridges across the yarn, ~1.1 mm along the leg
                ph = 2 * np.pi * (a / (1.15 * W / 6.0) + side * 0.9 * u + leg_phase[rr, cc, k])
                pl = 0.5 + 0.5 * np.cos(ph)
                hh = np.where(h > 0, h * (0.86 + 0.14 * pl), h)
                m = hh > hbest
                hbest = np.where(m, hh, hbest)
                ply = np.where(m, pl, ply)
                across = np.where(m, u, across)
                sval = np.where(m, stitch_val[rr, cc], sval)
                lval = np.where(m, leg_val[rr, cc, k], lval)
                is_purl = np.where(m, False, is_purl)
            if col_kind is not None:
                # purl bumps: horizontal bar at the row centre + smaller nubs between rows
                for (cy, ax, by, amp, off) in ((0.0, 0.56, 0.36, 1.0, 0.0), (0.5, 0.30, 0.22, 0.65, 0.5)):
                    bx = lx - off * W
                    by_ = ly - cy * H
                    aa = ax * W * (1 + 0.08 * stitch_val[rr, cc])
                    d2 = (bx / aa) ** 2 + (by_ / (by * H)) ** 2
                    h = np.sqrt(np.clip(1 - d2, 0, None)) * h_amp * 0.85 * amp
                    # purl bar is a yarn seen sideways: plies run along x
                    ph = 2 * np.pi * (bx / (1.1 * W / 5.0) + 0.9 * by_ / (by * H) + leg_phase[rr, cc, 0])
                    pl = 0.5 + 0.5 * np.cos(ph)
                    hh = np.where(kind == 1, h * (0.82 + 0.18 * pl) - purl_drop * (h > 0), 0.0)
                    hh = np.where(h > 0, hh, -1e9)
                    m = hh > hbest
                    hbest = np.where(m, hh, hbest)
                    ply = np.where(m, pl, ply)
                    across = np.where(m, by_ / (by * H), across)
                    sval = np.where(m, stitch_val[rr, cc], sval)
                    lval = np.where(m, leg_val[rr, cc, 0], lval)
                    is_purl = np.where(m, True, is_purl)
    return dict(h=hbest, ply=ply, across=across, sval=sval, lval=lval, purl=is_purl)


def _knit_material(res, tile_mm, ncol, nrow, seed, rib=False):
    rng = np.random.default_rng(seed)
    W, H = tile_mm / ncol, tile_mm / nrow
    x, y = _grid(res, tile_mm)
    px = tile_mm / res
    h_amp = 1.6 * W / 6.0
    R = 0.255 * W
    col_kind = None
    purl_drop = 0.0
    if rib:
        col_kind = np.array([0 if (i % 4) < 2 else 1 for i in range(ncol)])
        purl_drop = 1.1 * h_amp
        # 2x2 rib pulls the knit wales together: squeeze the knit pairs, open the purl troughs
        cidx = x / W
        k = np.floor(cidx).astype(int)
        f = cidx - k
        pair = (k % 4)
        shift = np.where(pair == 0, 0.10, np.where(pair == 1, -0.10, 0.0)) * W
        x = x + shift * np.sin(np.pi * f) ** 2
    st = _stitch_field(x, y, W, H, ncol, nrow, rng, h_amp, R, col_kind, purl_drop)
    h = st["h"]
    floor = -1.4 * h_amp if rib else -0.35 * h_amp
    h = np.maximum(h, floor) if rib else np.maximum(h, 0.0)
    if rib:
        h = np.where(h <= -1e8, floor, h)
    # fuzz / halo: fine fibres, plus a soft low-lying haze that fills the deepest gaps a bit
    fuzz = _noise(rng, res, 0.9) * 0.6 + _noise(rng, res, 2.5) * 0.4
    open_ = np.clip(h / h_amp, 0, 1)
    h = h + 0.045 * h_amp * fuzz * (0.4 + open_)
    h = _blur(h, 0.6)
    lo = h.min()
    nrm = _normal_from_height(h, px, strength=0.85)
    ao = _cavity_ao(h, px, (0.6 * W / 6, 1.6 * W / 6, 3.0 * W / 6), depth_mm=0.9 * h_amp, floor=0.2)
    # albedo
    base = _srgb_to_lin((196, 142, 52))
    # hue/value jitter per stitch and per yarn leg (a slightly heathered, hand-dyed yarn)
    v = 1 + 0.055 * np.tanh(st["sval"]) + 0.035 * np.tanh(st["lval"])
    hue = 0.035 * np.tanh(st["sval"] * 0.7 + st["lval"] * 0.5)
    col = np.stack([base[0] * (1 + hue), base[1] * (1 - 0.2 * hue), base[2] * (1 - 1.2 * hue)], -1)
    # long-wave dye unevenness so the 2x2 repeat doesn't read
    dye = _noise(rng, res, res / 18)
    col = col * (v * (1 + 0.04 * dye))[..., None]
    # ply grooves darker, ply crowns a touch lighter, fibre speckle
    plyshade = 0.86 + 0.20 * st["ply"]
    edge = 1 - 0.30 * np.abs(st["across"]) ** 3
    speck = 1 + 0.06 * _noise(rng, res, 0.7)
    col = col * (plyshade * edge * speck)[..., None]
    # grooves between stitches: dark
    depth = np.clip((h - lo) / (h_amp - lo), 0, 1)
    groove = 0.30 + 0.70 * np.clip(depth * 1.6, 0, 1) ** 0.8
    col = col * (groove * (0.55 + 0.45 * ao))[..., None]
    rough = np.clip(0.90 + 0.04 * fuzz - 0.03 * st["ply"], 0.82, 0.97)
    return col, nrm, ao, rough


# ----------------------------------------------------------------------------- weaves


def _weave(res, tile_mm, n_warp, n_weft, up, rng, thread_w=0.92, lift=1.0):
    """Generic woven height field. up[j, i] = True where warp (vertical thread i) floats over
    weft (horizontal thread j). Returns height (thread-diameter units), which-thread mask, ids."""
    x, y = _grid(res, tile_mm)
    sx, sy = tile_mm / n_warp, tile_mm / n_weft
    fi = x / sx
    fj = y / sy
    i = np.floor(fi).astype(int) % n_warp
    j = np.floor(fj).astype(int) % n_weft
    fx = fi - np.floor(fi) - 0.5          # across warp, -0.5..0.5
    fy = fj - np.floor(fj) - 0.5          # across weft
    wthick = 1 + 0.10 * rng.standard_normal(n_warp)
    fthick = 1 + 0.10 * rng.standard_normal(n_weft)
    upf = up.astype(np.float64)
    # elevation of the warp along y: interpolate float state between cell centres (cosine)
    jn = np.where(fy >= 0, (j + 1) % n_weft, (j - 1) % n_weft)
    tw = np.abs(fy)
    sm = 0.5 - 0.5 * np.cos(np.pi * tw)      # 0 at centre, 1 at boundary
    e_here = upf[j, i]
    e_nb = upf[jn, i]
    warp_elev = e_here + (0.5 * (e_here + e_nb) - e_here) * sm
    # weft elevation along x
    inb = np.where(fx >= 0, (i + 1) % n_warp, (i - 1) % n_warp)
    sm2 = 0.5 - 0.5 * np.cos(np.pi * np.abs(fx))
    f_here = 1 - upf[j, i]
    f_nb = 1 - upf[j, inb]
    weft_elev = f_here + (0.5 * (f_here + f_nb) - f_here) * sm2
    wprof = np.sqrt(np.clip(1 - (fx / (0.5 * thread_w * wthick[i])) ** 2, 0, None))
    fprof = np.sqrt(np.clip(1 - (fy / (0.5 * thread_w * fthick[j])) ** 2, 0, None))
    hw = wprof * (0.35 + lift * 0.65 * warp_elev)
    hf = fprof * (0.35 + lift * 0.65 * weft_elev)
    is_warp = hw >= hf
    h = np.maximum(hw, hf)
    return dict(h=h, warp=is_warp, i=i, j=j, x=x, y=y, fx=fx, fy=fy)


def _blouse(res, tile_mm, seed):
    rng = np.random.default_rng(seed)
    n = 72                                   # 0.28 mm threads over 20 mm
    ii, jj = np.meshgrid(np.arange(n), np.arange(n))
    up = ((ii + jj) % 2) == 0
    wv = _weave(res, tile_mm, n, n, up, rng, thread_w=0.95, lift=0.8)
    px = tile_mm / res
    h = _blur(wv["h"], 0.5) * 0.08          # mm; thread ~0.28, keep shallow
    nrm = _normal_from_height(h, px, strength=0.45)
    ao = _cavity_ao(h, px, (0.15, 0.4), depth_mm=0.05, floor=0.75)
    base = _srgb_to_lin((238, 228, 206))
    slub_w = 1 + 0.018 * rng.standard_normal(n)
    slub_f = 1 + 0.018 * rng.standard_normal(n)
    v = np.where(wv["warp"], slub_w[wv["i"]], slub_f[wv["j"]])
    v = v * (1 + 0.012 * _noise(rng, res, 1.5)) * (0.93 + 0.07 * np.clip(wv["h"], 0, 1))
    col = base[None, None, :] * v[..., None]
    rough = np.full((res, res), 0.88) + 0.02 * _noise(rng, res, 2)
    return col, nrm, ao, np.clip(rough, 0.8, 0.95)


def _trousers(res, tile_mm, seed):
    rng = np.random.default_rng(seed)
    n = 90                                   # 0.333 mm threads; 2/1 twill pitch 3*s/sqrt2 = 0.71 mm
    ii, jj = np.meshgrid(np.arange(n), np.arange(n))
    # warp over 2, under 1; diagonal rises to the right (ribs run lower-left -> upper-right)
    up = ((ii + jj) % 3) != 0
    wv = _weave(res, tile_mm, n, n, up, rng, thread_w=0.98, lift=1.0)
    px = tile_mm / res
    h = _blur(wv["h"], 0.8) * 0.16
    nrm = _normal_from_height(h, px, strength=0.6)
    ao = _cavity_ao(h, px, (0.2, 0.5), depth_mm=0.08, floor=0.6)
    warp_c = _srgb_to_lin((34, 78, 82))
    weft_c = _srgb_to_lin((28, 60, 64))
    # heathered threads: each warp varies along its length, per-thread tone offsets
    tw = 1 + 0.07 * rng.standard_normal(n)
    tf = 1 + 0.05 * rng.standard_normal(n)
    heath = _noise(rng, res, 3.0, aniso=(1.0, 0.12))        # streaks along warp (vertical)
    heath_f = _noise(rng, res, 3.0, aniso=(0.12, 1.0))
    vw = tw[wv["i"]] * (1 + 0.06 * heath)
    vf = tf[wv["j"]] * (1 + 0.05 * heath_f)
    col = np.where(wv["warp"][..., None], warp_c * vw[..., None], weft_c * vf[..., None])
    col = col * (0.80 + 0.20 * np.clip(wv["h"], 0, 1))[..., None]
    col = col * (1 + 0.03 * _noise(rng, res, res / 16))[..., None]
    rough = np.clip(0.86 + 0.03 * _noise(rng, res, 2), 0.8, 0.93)
    return col, nrm, ao, rough


# ----------------------------------------------------------------------------- leather / buttons


def _leather(res, tile_mm, seed):
    from scipy.spatial import cKDTree
    rng = np.random.default_rng(seed)
    px = tile_mm / res
    x, y = _grid(res, tile_mm)
    npts = int((tile_mm / 0.75) ** 2)        # ~0.75 mm pebbles
    pts = rng.uniform(0, tile_mm, (npts, 2))
    tree = cKDTree(pts, boxsize=tile_mm)
    q = np.stack([x.ravel() % tile_mm, y.ravel() % tile_mm], -1)
    d, _ = tree.query(q, k=2)
    f1 = d[:, 0].reshape(res, res)
    f2 = d[:, 1].reshape(res, res)
    crease = np.clip((f2 - f1) / 0.18, 0, 1)        # 0 on cell borders
    pebble = np.sqrt(crease) - 0.25 * (f1 / 0.5) ** 2
    fine = _noise(rng, res, 1.2)
    big = _noise(rng, res, res / 10)
    h = (0.025 * pebble + 0.004 * fine) * (1 + 0.25 * big)
    h = _blur(h, 0.7)
    nrm = _normal_from_height(h, px, strength=1.0)
    ao = _cavity_ao(h, px, (0.2, 0.5), depth_mm=0.02, floor=0.6)
    base = _srgb_to_lin((48, 29, 20))
    v = (1 + 0.10 * big + 0.04 * _noise(rng, res, 6)) * (0.80 + 0.20 * crease)
    col = base[None, None, :] * v[..., None]
    rough = np.clip(0.40 + 0.08 * (1 - crease) * 0 + 0.10 * (1 - crease) + 0.03 * fine, 0.33, 0.52)
    return col, nrm, ao, rough


def _buttons(res, tile_mm, seed):
    rng = np.random.default_rng(seed)
    px = tile_mm / res
    x, y = _grid(res, tile_mm)
    # torus distance from the tile centre -> concentric growth rings (continuous across the wrap)
    dx = np.abs(x - tile_mm / 2)
    dy = np.abs(y - tile_mm / 2)
    dx = np.minimum(dx, tile_mm - dx)
    dy = np.minimum(dy, tile_mm - dy)
    rr = np.hypot(dx, dy)
    warp = _noise(rng, res, res / 24)
    rings = 0.5 + 0.5 * np.sin(2 * np.pi * (rr / 0.9 + 0.35 * warp))
    streak = _noise(rng, res, 2.5, aniso=(0.08, 1.0))        # fine fibrous streaks
    blot = _noise(rng, res, res / 12)
    dark = _srgb_to_lin((52, 30, 18))
    light = _srgb_to_lin((112, 70, 40))
    t = np.clip(0.35 + 0.18 * rings + 0.14 * streak + 0.18 * blot, 0, 1)
    col = dark[None, None, :] * (1 - t[..., None]) + light[None, None, :] * t[..., None]
    h = 0.003 * streak + 0.002 * rings
    nrm = _normal_from_height(h, px, strength=1.0)
    ao = np.full((res, res), 1.0)
    rough = np.clip(0.30 + 0.04 * streak, 0.24, 0.38)
    return col, nrm, ao, rough


# ----------------------------------------------------------------------------- entry


WASH = {"cloth_knit": (196, 134, 42), "cloth_rib": (186, 124, 38), "cloth_blouse": (240, 231, 210),
        "cloth_trousers": (26, 80, 86)}


def _wash(res, base_srgb, seed, sig_frac=0.12, amp=0.07):
    """stitch-free albedo: base colour + large-scale dye wash (the shader weave supplies stitches / relief)."""
    rng = np.random.default_rng(seed)
    n = _noise(rng, res, res * sig_frac)
    base = _srgb_to_lin(base_srgb)
    col = base[None, None, :] * (1 + amp * n)[..., None]
    nrm = np.dstack([np.full((res, res), 128), np.full((res, res), 128), np.full((res, res), 255)]).astype(np.uint8)
    ao = np.ones((res, res))
    rough = np.full((res, res), 0.9)
    return col, nrm, ao, rough


# ----------------------------------------------------------------------------- junor knit / rib / poplin


def _warp_coords(rng, x, y, res, tile_mm, W, H, amp=(0.14, 0.11), corr_mm=22.0):
    """tileable domain warp of the stitch grid: rows wander, columns lean, stitches vary in size (hand knitting)."""
    px = tile_mm / res
    n1 = _noise(rng, res, corr_mm / px) + 0.45 * _noise(rng, res, 0.4 * corr_mm / px)
    n2 = _noise(rng, res, corr_mm / px) + 0.45 * _noise(rng, res, 0.4 * corr_mm / px)
    return x + amp[0] * W * n1, y + amp[1] * H * n2


def _knit_r(res, tile_mm, ncol, nrow, seed, rib=False, base_srgb=(218, 169, 40)):
    # r5 (SHADE: the knit read orange): base chosen so the map MEAN is ~sRGB (200, 155, 40) mustard (hue ~44 deg)
    rng = np.random.default_rng(seed)
    W, H = tile_mm / ncol, tile_mm / nrow
    x, y = _grid(res, tile_mm)
    px = tile_mm / res
    h_amp = 1.6 * W / 6.0
    R = 0.26 * W
    col_kind = None
    purl_drop = 0.0
    if rib:
        col_kind = np.array([i % 2 for i in range(ncol)])           # 1x1 rib: knit, purl, knit, purl
        purl_drop = 0.9 * h_amp
    xw, yw = _warp_coords(rng, x, y, res, tile_mm, W, H, amp=(0.10, 0.08) if rib else (0.14, 0.11))
    if rib:
        # 1x1 rib pulls the knit wales together over the purls: knit column 1.3 W wide, purl trough 0.7 W
        t = xw / (2 * W)
        k = np.floor(t)
        f = t - k
        g = np.where(f < 0.65, f / 0.65 * 0.5, 0.5 + (f - 0.65) / 0.35 * 0.5)
        xw = (k + g) * 2 * W
    st = _stitch_field(xw, yw, W, H, ncol, nrow, rng, h_amp, R, col_kind, purl_drop)
    h = st["h"]
    if rib:
        floor = -1.3 * h_amp
        h = np.where(h <= -1e8, floor, np.maximum(h, floor))
    else:
        h = np.maximum(h, 0.0)
    # per-row tension (some rows tighter / flatter) + a low-frequency puckering of the fabric
    tens = 1 + 0.10 * _noise(rng, res, 8.0 / px, aniso=(0.15, 1.0))
    h = h * tens + 0.12 * h_amp * _noise(rng, res, 30.0 / px)
    # fibre fuzz: fine fibres everywhere, stronger on the yarn crowns, and a soft halo that half-fills the gaps
    fz = _noise(rng, res, 0.7) * 0.55 + _noise(rng, res, 1.8) * 0.45
    open_ = np.clip(h / h_amp, 0, 1)
    halo = np.clip(_blur(np.clip(h, 0, None), 2.2 / px) - h, 0, None)
    h = h + 0.06 * h_amp * fz * (0.3 + open_) + 0.35 * halo
    h = _blur(h, 0.6)
    lo = h.min()
    nrm = _normal_from_height(h, px, strength=0.9)
    ao = _cavity_ao(h, px, (0.6 * W / 6, 1.6 * W / 6, 3.0 * W / 6), depth_mm=0.9 * h_amp, floor=0.22)
    base = _srgb_to_lin(base_srgb)
    v = 1 + 0.06 * np.tanh(st["sval"]) + 0.04 * np.tanh(st["lval"])
    hue = 0.04 * np.tanh(st["sval"] * 0.7 + st["lval"] * 0.5)
    col = np.stack([base[0] * (1 + hue), base[1] * (1 - 0.2 * hue), base[2] * (1 - 1.2 * hue)], -1)
    # two low-frequency variations (no visible repeat at bust framing): dye lot wash ~6 cm and a ~2 cm mottling
    lf = 0.06 * _noise(rng, res, 60.0 / px) + 0.035 * _noise(rng, res, 20.0 / px)
    col = col * (v * (1 + lf))[..., None]
    # ply twist: grooves darker, crowns lighter; yarn edges roll away; fibre speckle
    plyshade = 0.84 + 0.22 * st["ply"]
    edge = 1 - 0.32 * np.abs(st["across"]) ** 3
    speck = 1 + 0.07 * _noise(rng, res, 0.7)
    col = col * (plyshade * edge * speck)[..., None]
    depth = np.clip((h - lo) / (h_amp - lo), 0, 1)
    groove = 0.32 + 0.68 * np.clip(depth * 1.6, 0, 1) ** 0.8
    col = col * (groove * (0.55 + 0.45 * ao))[..., None]
    # fibre halo: a slightly paler, desaturated haze over the gaps and stitch edges (wool/acrylic blend)
    hz = np.clip(halo / (0.25 * h_amp), 0, 1) * (0.5 + 0.5 * np.clip(fz, 0, 1))
    grey = col.mean(-1, keepdims=True)
    col = col * (1 - 0.35 * hz[..., None]) + (0.7 * col + 0.3 * grey) * 1.12 * (0.35 * hz[..., None])
    rough = np.clip(0.90 + 0.04 * fz - 0.03 * st["ply"], 0.82, 0.97)
    return col, nrm, ao, rough


def _poplin(res, tile_mm, seed):
    """cotton poplin: fine warp, slightly thicker weft (faint horizontal rib), slubs + a soft cotton mottle."""
    rng = np.random.default_rng(seed)
    nw, nf = 168, 120                         # 0.24 mm warp, 0.33 mm weft over 40 mm
    ii, jj = np.meshgrid(np.arange(nw), np.arange(nf))
    up = ((ii + jj) % 2) == 0
    wv = _weave(res, tile_mm, nw, nf, up, rng, thread_w=0.96, lift=0.7)
    px = tile_mm / res
    h = _blur(wv["h"], 0.5) * 0.05 + 0.006 * _noise(rng, res, 6.0 / px)
    nrm = _normal_from_height(h, px, strength=0.4)
    ao = _cavity_ao(h, px, (0.15, 0.4), depth_mm=0.04, floor=0.8)
    base = _srgb_to_lin((241, 236, 224))
    slub_w = 1 + 0.02 * rng.standard_normal(nw)
    slub_f = 1 + 0.025 * rng.standard_normal(nf)
    v = np.where(wv["warp"], slub_w[wv["i"]], slub_f[wv["j"]])
    v = v * (1 + 0.012 * _noise(rng, res, 1.5) + 0.015 * _noise(rng, res, 8.0 / px)) * (0.94 + 0.06 * np.clip(wv["h"], 0, 1))
    col = base[None, None, :] * v[..., None]
    rough = np.clip(0.86 + 0.02 * _noise(rng, res, 2), 0.8, 0.93)
    return col, nrm, ao, rough


def make_textures_r(texdir, res=1024, knit_res=2048):
    """junor: baked knit / 1x1 rib (shader weave off) + poplin + twill + juno's leather / buttons."""
    specs = [
        ("cloth_knit", "knit", lambda r: _knit_r(knit_res, TILE_M["knit"] * 1e3, 36, 48, 11), [0.42, 0.30, 0.13], 0.55, 0.35, {}),
        ("cloth_rib", "rib", lambda r: _knit_r(r, TILE_M["rib"] * 1e3, 30, 32, 23, rib=True, base_srgb=(210, 160, 35)),
         [0.42, 0.30, 0.13], 0.55, 0.35, {}),
        ("cloth_blouse", "blouse", lambda r: _poplin(r, TILE_M["blouse"] * 1e3, 31), [0.95, 0.93, 0.88], 0.45, 0.1, {}),
        ("cloth_trousers", "trousers", lambda r: _trousers(r, TILE_M["trousers"] * 1e3, 41), [0.20, 0.38, 0.40], 0.50, 0.1, {}),
        ("shoes", "leather", lambda r: _leather(r, 50.0, 53), [0.30, 0.16, 0.13], 0.35, 0.1, {}),
        ("cloth_buttons", "buttons", lambda r: _buttons(min(r, 256), 20.0, 61), [0.25, 0.16, 0.10], 0.30, 0.1,
         {"type": "buttons"}),
    ]
    out = {}
    for mid, kind, fn, sheen, sheen_r, thick, extra in specs:
        col, nrm, ao, rough = fn(res)
        _save(texdir, mid, col, nrm, ao, rough, 0.5, thick)
        typ = extra.get("type", "shoes" if mid == "shoes" else "cloth")
        out[mid] = {"type": typ, "albedo": f"tex/{mid}_albedo.png", "normal": f"tex/{mid}_normal.png",
                    "packed": f"tex/{mid}_packed.png",
                    "params": {"sheen_color": list(sheen), "sheen_roughness": sheen_r,
                               "tile_m": TILE_M[kind], "uv": "metres/tile_m, wrap repeat"}}
    return out


def make_textures(texdir, res=1024, shader_weave=None) -> dict:
    """Write all wardrobe textures into texdir; return the material-json dict (paths 'tex/...').
    shader_weave (default: env WARDROBE_BAKED != 1): cloth maps are stitch-free colour wash only (RENDER-cloth's shader
    makes the stitches); WARDROBE_BAKED=1 bakes the stitch relief into the maps instead (for Blender previews)."""
    res = int(res)
    if shader_weave is None:
        shader_weave = os.environ.get("WARDROBE_BAKED", "0") != "1"
    specs = [
        # id, kind, builder, sheen colour (linear), sheen roughness, thickness, extra
        ("cloth_knit", "knit", lambda r: _knit_material(r, 120.0, 20, 24, 11), [0.42, 0.30, 0.13], 0.55, 0.35, {}),
        ("cloth_rib", "rib", lambda r: _knit_material(r, 60.0, 12, 14, 23, rib=True), [0.42, 0.30, 0.13], 0.55, 0.35, {}),
        ("cloth_blouse", "blouse", lambda r: _blouse(r, 20.0, 31), [0.95, 0.92, 0.86], 0.45, 0.1, {}),
        ("cloth_trousers", "trousers", lambda r: _trousers(r, 30.0, 41), [0.20, 0.38, 0.40], 0.50, 0.1, {}),
        ("shoes", "leather", lambda r: _leather(r, 50.0, 53), [0.30, 0.16, 0.13], 0.35, 0.1, {}),
        ("cloth_buttons", "buttons", lambda r: _buttons(min(r, 256), 20.0, 61), [0.25, 0.16, 0.10], 0.30, 0.1,
         {"type": "buttons"}),
    ]
    out = {}
    for mid, kind, fn, sheen, sheen_r, thick, extra in specs:
        if shader_weave and mid in WASH:
            col, nrm, ao, rough = _wash(min(res, 256), WASH[mid], 7 + len(mid))
        else:
            col, nrm, ao, rough = fn(res)
        _save(texdir, mid, col, nrm, ao, rough, 0.5, thick)
        typ = extra.get("type", "shoes" if mid == "shoes" else "cloth")
        out[mid] = {"type": typ, "albedo": f"tex/{mid}_albedo.png", "normal": f"tex/{mid}_normal.png",
                    "packed": f"tex/{mid}_packed.png",
                    "params": {"sheen_color": list(sheen), "sheen_roughness": sheen_r,
                               "tile_m": TILE_M[kind], "uv": "metres/tile_m, wrap repeat"}}
    return out


if __name__ == "__main__":
    import json
    import sys
    import time
    d = sys.argv[1] if len(sys.argv) > 1 else "tex"
    t0 = time.time()
    m = make_textures(d, int(sys.argv[2]) if len(sys.argv) > 2 else 1024)
    print(json.dumps(m, indent=1))
    print(f"{time.time() - t0:.1f}s")
